#!/usr/bin/env python3
"""Build reciprocal-descriptor/union-find tracks without COLMAP points.

The existing v6 pointmap graph remains a valid geometry source, but its
projection-star correspondences cover only the Chart cameras.  This producer
selects calibrated keyframes across the complete Cambridge database, runs
MASt3R reciprocal descriptor matching, gates every edge with exact K and
fixed-pose triangulation, merges observation nodes with union-find, and
exports the observation-level archive consumed by the unified Teacher.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MAST3R_ROOT = ROOT / "mast3r"
for path in (ROOT, MAST3R_ROOT, MAST3R_ROOT / "dust3r"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.inference import inference  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402
from mast3r.fast_nn import fast_reciprocal_NNs  # noqa: E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from outdoor.mast3r_track_graph import (  # noqa: E402
    CameraUniqueUnionFind,
    ROLE_NAMES,
    ROLE_CANOPY,
    ROLE_RIGID,
    TRACK_GRAPH_VERSION,
    _project,
    _roles_at,
    _sequence,
    _triangulate,
    descriptor_cycle_consistency,
    descriptor_view_direction_cost,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_keyframes(records: list[dict], maximum: int) -> list[dict]:
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_sequence[str(row["sequence_id"])].append(row)
    for rows in by_sequence.values():
        rows.sort(key=lambda value: int(value["frame_index"]))
    total = sum(map(len, by_sequence.values()))
    if maximum <= 0 or maximum >= total:
        return sorted(records, key=lambda row: int(row["frame_index"]))
    selected = []
    for sequence in sorted(by_sequence):
        rows = by_sequence[sequence]
        quota = max(2, int(round(maximum * len(rows) / total)))
        indices = np.linspace(
            0, len(rows) - 1, min(quota, len(rows)), dtype=np.int64
        )
        selected.extend(rows[int(index)] for index in np.unique(indices))
    selected.sort(key=lambda row: int(row["frame_index"]))
    if len(selected) > maximum:
        indices = np.linspace(
            0, len(selected) - 1, maximum, dtype=np.int64
        )
        selected = [selected[int(index)] for index in indices]
    return selected


def _pair_graph(
    records: list[dict],
    *,
    temporal_neighbours: int,
    cross_sequence_neighbours: int,
) -> list[tuple[int, int]]:
    centers = np.asarray(
        [row["camera_center_world"] for row in records], dtype=np.float64
    )
    c2w = np.asarray(
        [row["T_camera_to_world"] for row in records], dtype=np.float64
    )
    forward = c2w[:, :3, 2]
    sequences = [str(row["sequence_id"]) for row in records]
    edges = set()
    for source in range(len(records)):
        same = [
            index
            for index in range(len(records))
            if index != source and sequences[index] == sequences[source]
        ]
        same.sort(
            key=lambda index: (
                abs(
                    int(records[index]["frame_index"])
                    - int(records[source]["frame_index"])
                ),
                index,
            )
        )
        other = np.asarray(
            [
                index
                for index in range(len(records))
                if sequences[index] != sequences[source]
            ],
            dtype=np.int64,
        )
        cross = []
        if len(other):
            distance = np.linalg.norm(
                centers[other] - centers[source], axis=1
            )
            # Opposite optical axes are not an overlapping view.  The
            # previous absolute dot product assigned the same orientation
            # cost to cameras looking in the same and opposite directions,
            # which admitted many geometrically impossible cross-sequence
            # pairs before the expensive descriptor pass.
            direction = descriptor_view_direction_cost(
                forward[source], forward[other]
            )
            score = distance + 4.0 * direction
            cross = other[np.argsort(score, kind="stable")][
                :cross_sequence_neighbours
            ].tolist()
        for target in same[:temporal_neighbours] + cross:
            edges.add(tuple(sorted((source, int(target)))))
    return sorted(edges)


def _camera_arrays(records: list[dict], image_root: Path):
    transforms = []
    intrinsics = []
    sizes = []
    paths = []
    for row in records:
        path = image_root / row["image_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        camera = row["camera"]
        sx = width / float(camera["width"])
        sy = height / float(camera["height"])
        intrinsics.append(
            [
                float(camera["fx"]) * sx,
                float(camera["fy"]) * sy,
                float(camera["cx"]) * sx,
                float(camera["cy"]) * sy,
            ]
        )
        transforms.append(row["T_camera_to_world"])
        sizes.append((width, height))
        paths.append(path)
    if len(set(sizes)) != 1:
        raise RuntimeError(
            "Descriptor track producer currently requires one RGB raster size"
        )
    return (
        paths,
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(transforms, dtype=np.float64),
        np.asarray(sizes, dtype=np.int32),
    )


def _role_maps(
    dataset: Path,
    tree_mask_pickle: Path | None,
    names: list[str],
):
    if tree_mask_pickle is None:
        return [None] * len(names)
    lookup = CambridgeMaskLookup(
        dataset, tree_mask_pickle, mask_indices=[0, 1, 2, 3]
    )
    maps = []
    for name in names:
        channels = lookup.masks[lookup.source_name_for(name)]
        maps.append(
            np.stack(
                [
                    channel.detach().cpu().bool().numpy()
                    for channel in channels[:4]
                ]
            )
        )
    return maps


def build(args) -> dict:
    # Snapshot provenance before any expensive inference.  Reading this hash
    # only while writing the final summary can mislabel an already-running
    # producer when the source file is edited during a long descriptor job.
    producer_implementation_sha256 = _sha256(Path(__file__).resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        # Dust3R/MASt3R contains a few legacy bare ``.cuda()`` allocations.
        # Make an indexed CLI device authoritative for those tensors too;
        # otherwise ``--device cuda:2`` mixes GPU 0 and GPU 2 in one pair.
        torch.cuda.set_device(
            0 if device.index is None else int(device.index)
        )
    scene = json.loads(args.scene_contract.read_text(encoding="utf-8"))
    records = _select_keyframes(
        list(scene["records"]), args.maximum_keyframes
    )
    image_root = args.dataset / "images"
    paths, intrinsics, c2w, sizes = _camera_arrays(records, image_root)
    width, height = map(int, sizes[0])
    # Robust component triangulation may emit tens of thousands of tracks.
    # Opening the same JPEG once per track made the post-MASt3R stage slower
    # than all descriptor inference and grew the process to >14 GB through
    # decoder churn.  The selected keyframe RGBs are only ~90 MB at 128x640p.
    rgb_images = []
    for path in paths:
        with Image.open(path) as image:
            rgb_images.append(np.asarray(image.convert("RGB")).copy())
    names = [Path(row["image_name"]).stem for row in records]
    sequences = [_sequence(name) for name in names]
    role_maps = _role_maps(args.dataset, args.tree_mask_pickle, names)
    pairs = _pair_graph(
        records,
        temporal_neighbours=args.temporal_neighbours,
        cross_sequence_neighbours=args.cross_sequence_neighbours,
    )
    model = AsymmetricMASt3R.from_pretrained(
        str(args.checkpoint)
    ).to(args.device).eval()
    union = CameraUniqueUnionFind()
    node_by_key: dict[tuple[int, int, int], int] = {}
    observation: list[dict] = []
    edge_count: dict[tuple[int, int], int] = defaultdict(int)

    def node(camera: int, pixel: np.ndarray, confidence: float) -> int:
        key = (
            int(camera),
            int(round(float(pixel[0]))),
            int(round(float(pixel[1]))),
        )
        index = node_by_key.get(key)
        if index is None:
            index = union.add(camera)
            node_by_key[key] = index
            observation.append(
                {
                    "camera": int(camera),
                    "pixel": np.asarray(pixel, dtype=np.float32),
                    "confidence": float(confidence),
                }
            )
        else:
            observation[index]["confidence"] = max(
                observation[index]["confidence"], float(confidence)
            )
        return index

    reciprocal_total = 0
    exact_k_accepted = 0
    for first, second in tqdm(pairs, desc="MASt3R descriptor pairs"):
        images = load_images(
            [str(paths[first]), str(paths[second])],
            size=512,
            verbose=False,
        )
        result = inference(
            [tuple(images)],
            model,
            args.device,
            batch_size=1,
            verbose=False,
        )
        desc_first = result["pred1"]["desc"].squeeze(0).detach()
        desc_second = result["pred2"]["desc"].squeeze(0).detach()
        match_first, match_second = fast_reciprocal_NNs(
            desc_first,
            desc_second,
            subsample_or_initxy1=args.subsample,
            device=args.device,
            dist="dot",
            block_size=2**13,
        )
        match_first = np.asarray(match_first)
        match_second = np.asarray(match_second)
        reciprocal_total += len(match_first)
        shape_first = np.asarray(
            result["view1"]["true_shape"][0].detach().cpu()
        )
        shape_second = np.asarray(
            result["view2"]["true_shape"][0].detach().cpu()
        )
        scale_first = np.asarray(
            [width / shape_first[1], height / shape_first[0]]
        )
        scale_second = np.asarray(
            [width / shape_second[1], height / shape_second[0]]
        )
        match_first = match_first * scale_first
        match_second = match_second * scale_second
        for pixel_first, pixel_second in zip(
            match_first, match_second
        ):
            pixels = np.stack([pixel_first, pixel_second]).astype(np.float32)
            camera_indices = np.asarray([first, second], dtype=np.int32)
            xyz, residual, median, geometry = _triangulate(
                pixels,
                camera_indices,
                c2w,
                intrinsics,
                width,
                height,
                np.ones(2, dtype=np.float32),
            )
            roles = np.asarray(
                [
                    _roles_at(
                        role_maps[camera],
                        pixels[slot : slot + 1],
                        width,
                        height,
                    )[0]
                    for slot, camera in enumerate(camera_indices)
                ],
                dtype=np.int8,
            )
            if (
                not np.isfinite(xyz).all()
                or median > args.maximum_reprojection_error
                or float(geometry[1]) < args.minimum_triangulation_angle
                or roles[0] != roles[1]
                or (
                    roles[0] not in {ROLE_RIGID, ROLE_CANOPY}
                    and sequences[first] != sequences[second]
                )
            ):
                continue
            first_node = node(first, pixel_first, 1.0 / (1.0 + residual[0]))
            second_node = node(
                second, pixel_second, 1.0 / (1.0 + residual[1])
            )
            if not union.union(first_node, second_node):
                continue
            edge_count[tuple(sorted((first_node, second_node)))] += 1
            exact_k_accepted += 1

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(observation)):
        components[union.find(index)].append(index)
    component_edge_count: dict[int, int] = defaultdict(int)
    for first_node, second_node in edge_count:
        root = union.find(first_node)
        if root != union.find(second_node):
            raise RuntimeError(
                "Union-find component contains an unmerged descriptor edge"
            )
        component_edge_count[root] += 1
    records_out = []
    for component_root, component in components.items():
        best_by_camera = {}
        for index in component:
            row = observation[index]
            old = best_by_camera.get(row["camera"])
            if old is None or row["confidence"] > old["confidence"]:
                best_by_camera[row["camera"]] = row
        rows = list(best_by_camera.values())
        if len(rows) < 2:
            continue
        cameras = np.asarray(
            [row["camera"] for row in rows], dtype=np.int32
        )
        pixels = np.stack([row["pixel"] for row in rows])
        confidence = np.asarray(
            [row["confidence"] for row in rows], dtype=np.float32
        )
        xyz, residual, median, geometry = _triangulate(
            pixels,
            cameras,
            c2w,
            intrinsics,
            width,
            height,
            np.clip(
                confidence / max(float(np.median(confidence)), 1e-6),
                0.25,
                4.0,
            ),
        )
        if (
            median > args.maximum_reprojection_error
            or geometry[1] < args.minimum_triangulation_angle
        ):
            continue
        # Every stored descriptor edge is reciprocal (A->B and B->A agree).
        # For components with at least three observation nodes, E-V+1 is the
        # number of additional independent graph cycles.  Combine that
        # topological closure with fixed-camera reprojection consistency
        # instead of exporting the former unconditional value of 1.0.
        descriptor_edges = int(component_edge_count[component_root])
        cycle_rank, cycle_consistency = descriptor_cycle_consistency(
            median,
            args.maximum_reprojection_error,
            descriptor_edges,
            len(component),
        )
        roles = np.concatenate(
            [
                _roles_at(
                    role_maps[int(camera)],
                    pixels[slot : slot + 1],
                    width,
                    height,
                )
                for slot, camera in enumerate(cameras)
            ]
        )
        role_count = np.bincount(
            roles, minlength=len(ROLE_NAMES)
        ).astype(np.int16)
        dominant = int(np.argmax(role_count))
        sequence_count = len({sequences[int(index)] for index in cameras})
        if dominant == ROLE_RIGID and sequence_count < 2:
            continue
        if (
            dominant == ROLE_CANOPY
            and sequence_count >= 2
            and (
                len(rows) < 3
                or median
                > min(
                    float(args.maximum_reprojection_error),
                    float(args.maximum_cross_sequence_canopy_reprojection_error),
                )
            )
        ):
            # A reciprocal two-ray match can always triangulate some point,
            # so it is not sufficient evidence that a moving leaf is one
            # persistent physical primitive.  Stable trunks/branches survive
            # a third calibrated observation and a stricter joint residual;
            # transient leaf coincidences remain sequence-local.
            continue
        depths = np.asarray(
            [
                _project(
                    xyz[None],
                    c2w[int(camera)],
                    intrinsics[int(camera)],
                    width,
                    height,
                )[2][0]
                for camera in cameras
            ],
            dtype=np.float32,
        )
        color = rgb_images[int(cameras[0])][
            np.clip(round(float(pixels[0, 1])), 0, height - 1),
            np.clip(round(float(pixels[0, 0])), 0, width - 1),
        ]
        records_out.append(
            {
                "xyz": xyz,
                "rgb": color,
                "cameras": cameras,
                "pixels": pixels,
                "confidence": confidence,
                "residual": residual,
                "depth": depths,
                "geometry": geometry,
                "role_count": role_count,
                "dominant": dominant,
                "sequence_count": sequence_count,
                "descriptor_edge_count": descriptor_edges,
                "cycle_rank": cycle_rank,
                "cycle_consistency": cycle_consistency,
            }
        )
    if not records_out:
        raise RuntimeError("No descriptor track survived exact-K validation")
    offsets = np.zeros(len(records_out) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        [len(row["cameras"]) for row in records_out]
    )
    role_count = np.stack([row["role_count"] for row in records_out])
    role_probability = role_count / role_count.sum(1, keepdims=True).clip(1)
    arrays = {
        "schema_version": np.asarray(TRACK_GRAPH_VERSION),
        "correspondence_builder": np.asarray(
            "mast3r_reciprocal_descriptor_union_find"
        ),
        "camera_scope": np.asarray(
            "database_keyframes_full_sequence_coverage"
        ),
        "track_id": np.arange(len(records_out), dtype=np.int64),
        "xyz": np.stack([row["xyz"] for row in records_out]).astype(np.float32),
        "rgb": np.stack([row["rgb"] for row in records_out]).astype(np.uint8),
        "reprojection_error": np.asarray(
            [np.median(row["residual"]) for row in records_out],
            dtype=np.float32,
        ),
        "valid_observation_count": np.asarray(
            [len(row["cameras"]) for row in records_out], dtype=np.int16
        ),
        "sequence_count": np.asarray(
            [row["sequence_count"] for row in records_out], dtype=np.int16
        ),
        "role_observation_count": role_count,
        "role_probabilities": role_probability.astype(np.float32),
        "dominant_role": np.asarray(
            [row["dominant"] for row in records_out], dtype=np.int8
        ),
        "position_covariance_diag": np.stack(
            [row["geometry"][3:] for row in records_out]
        ).astype(np.float32),
        "triangulation_angle_p10": np.asarray(
            [row["geometry"][0] for row in records_out], dtype=np.float32
        ),
        "triangulation_angle_median": np.asarray(
            [row["geometry"][1] for row in records_out], dtype=np.float32
        ),
        "triangulation_angle_p90": np.asarray(
            [row["geometry"][2] for row in records_out], dtype=np.float32
        ),
        "cycle_consistency": np.asarray(
            [row["cycle_consistency"] for row in records_out],
            dtype=np.float32,
        ),
        "descriptor_edge_count": np.asarray(
            [row["descriptor_edge_count"] for row in records_out],
            dtype=np.int32,
        ),
        "descriptor_cycle_rank": np.asarray(
            [row["cycle_rank"] for row in records_out],
            dtype=np.int16,
        ),
        "observation_offsets": offsets,
        "observation_camera_indices": np.concatenate(
            [row["cameras"] for row in records_out]
        ).astype(np.int32),
        "observation_pixels": np.concatenate(
            [row["pixels"] for row in records_out]
        ).astype(np.float32),
        "observation_confidence": np.concatenate(
            [row["confidence"] for row in records_out]
        ).astype(np.float32),
        "observation_reprojection_error": np.concatenate(
            [row["residual"] for row in records_out]
        ).astype(np.float32),
        "observation_camera_depth": np.concatenate(
            [row["depth"] for row in records_out]
        ).astype(np.float32),
        "observation_centered_reprojection_error": np.concatenate(
            [row["residual"] for row in records_out]
        ).astype(np.float32),
        "camera_names": np.asarray(names),
        "camera_image_sizes": sizes,
        "support_camera_offsets": offsets,
        "support_camera_ids": np.concatenate(
            [row["cameras"] for row in records_out]
        ).astype(np.int32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "schema_version": TRACK_GRAPH_VERSION,
        "correspondence_builder": (
            "mast3r_reciprocal_descriptor_union_find"
        ),
        "camera_scope": "database_keyframes_full_sequence_coverage",
        "database_camera_count": int(scene["image_count"]),
        "selected_keyframe_count": len(records),
        "pair_count": len(pairs),
        "reciprocal_match_count": int(reciprocal_total),
        "exact_k_accepted_edge_count": int(exact_k_accepted),
        "exact_k_accepted_fraction": float(
            exact_k_accepted / max(reciprocal_total, 1)
        ),
        "track_count": len(records_out),
        "observation_count": int(offsets[-1]),
        "cross_sequence_track_count": int(
            (arrays["sequence_count"] >= 2).sum()
        ),
        "cross_sequence_canopy_track_count": int(
            (
                (arrays["sequence_count"] >= 2)
                & (arrays["dominant_role"] == ROLE_CANOPY)
            ).sum()
        ),
        "cross_sequence_canopy_contract": (
            "minimum_three_fixed_camera_observations__"
            "strict_joint_reprojection"
        ),
        "graph_cycle_track_count": int(
            (arrays["descriptor_cycle_rank"] > 0).sum()
        ),
        "graph_cycle_track_fraction": float(
            (arrays["descriptor_cycle_rank"] > 0).mean()
        ),
        "cycle_consistency_percentiles": np.percentile(
            arrays["cycle_consistency"], [10, 50, 90, 99]
        ).tolist(),
        "reprojection_error_percentiles": np.percentile(
            arrays["reprojection_error"], [50, 90, 99]
        ).tolist(),
        "selected_sequence_count": len(set(sequences)),
        "selected_keyframe_database_fraction": float(
            len(records) / max(int(scene["image_count"]), 1)
        ),
        "points3D_bin_read": False,
        "colmap_tracks_read": False,
        "scene_contract_sha256": _sha256(args.scene_contract),
        "tree_mask_pickle_sha256": (
            _sha256(args.tree_mask_pickle)
            if args.tree_mask_pickle is not None
            else None
        ),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "producer_implementation_sha256": (
            producer_implementation_sha256
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "mast3r/checkpoints/"
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-keyframes", type=int, default=256)
    parser.add_argument("--temporal-neighbours", type=int, default=2)
    parser.add_argument("--cross-sequence-neighbours", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--maximum-reprojection-error", type=float, default=1.5)
    parser.add_argument(
        "--maximum-cross-sequence-canopy-reprojection-error",
        type=float,
        default=0.75,
    )
    parser.add_argument("--minimum-triangulation-angle", type=float, default=0.35)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
