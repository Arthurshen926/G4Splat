#!/usr/bin/env python
"""Close static-skeleton and sequence-local lineage gaps in a foliage seed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import sha256_file  # noqa: E402
from outdoor.foliage_geometry import (  # noqa: E402
    evidence_conditioned_leaf_optical_mass,
    measured_static_skeleton_geometry,
)


PROTOCOL = "foliage-seed-contract-repair-v2-evidence-optical-mass"
LAYER_CANONICAL_CROWN = 0
LAYER_STATIC_SKELETON = 1
LAYER_DYNAMIC_LEAF = 2
SOURCE_DENSE_RAY = 4
_FRAME_PATTERN = re.compile(r"(?:frame)?(?P<frame>\d+)(?=\D*$)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skeleton-voxel-size", type=float, default=0.12)
    parser.add_argument(
        "--skeleton-minimum-confidence", type=float, default=0.18
    )
    parser.add_argument(
        "--sequence-merge-radius", type=float, default=0.06
    )
    parser.add_argument(
        "--sequence-merge-maximum-frame-gap", type=int, default=8
    )
    parser.add_argument(
        "--sequence-merge-maximum-group-size", type=int, default=3
    )
    return parser.parse_args()


def _camera_time_lookup(audit: dict) -> dict[int, tuple[str, int]]:
    result = {}
    for row in audit.get("selected_views", []):
        name = str(row.get("image_name", ""))
        match = _FRAME_PATTERN.search(name)
        frame = (
            int(match.group("frame"))
            if match
            else int(row["image_id"])
        )
        result[int(row["image_id"])] = (
            str(row.get("sequence_id", name.split("__", 1)[0])),
            frame,
        )
    return result


def _primary_observation_camera(payload: dict) -> np.ndarray:
    observations = payload["observation_camera_ids"].numpy()
    support = payload["support_camera_ids"].numpy()
    result = np.full(len(observations), -1, dtype=np.int32)
    if observations.shape[1]:
        valid = observations >= 0
        has = valid.any(axis=1)
        result[has] = observations[
            np.arange(len(observations))[has],
            valid.argmax(axis=1)[has],
        ]
    if support.shape[1]:
        missing = result < 0
        valid = support >= 0
        has = missing & valid.any(axis=1)
        result[has] = support[
            np.arange(len(support))[has],
            valid.argmax(axis=1)[has],
        ]
    return result


def sequence_local_correspondence_groups(
    centers: np.ndarray,
    tree_instance_id: np.ndarray,
    camera_ids: np.ndarray,
    camera_time: dict[int, tuple[str, int]],
    eligible: np.ndarray,
    *,
    radius: float = 0.06,
    maximum_frame_gap: int = 8,
    maximum_group_size: int = 3,
) -> list[np.ndarray]:
    """Build conservative reciprocal cross-frame leaf correspondences.

    Same-camera neighbours are never merged. Candidate edges are mutual
    nearest neighbours in world space, and union-find rejects a component if
    it would contain the same camera twice, exceed the temporal span, or
    exceed the observation-slot capacity.
    """
    xyz = np.asarray(centers, dtype=np.float64)
    instance = np.asarray(tree_instance_id, dtype=np.int32).reshape(-1)
    cameras = np.asarray(camera_ids, dtype=np.int32).reshape(-1)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    if not (
        len(xyz) == len(instance) == len(cameras) == len(eligible)
    ):
        raise ValueError("sequence correspondence inputs must have equal rows")
    if radius <= 0 or maximum_frame_gap < 0 or maximum_group_size < 2:
        raise ValueError("invalid sequence-local merge controls")

    rows_by_sequence_instance_camera = defaultdict(list)
    temporal = {}
    for row in np.flatnonzero(eligible):
        camera = int(cameras[row])
        if camera not in camera_time or int(instance[row]) < 0:
            continue
        sequence, frame = camera_time[camera]
        rows_by_sequence_instance_camera[
            (sequence, int(instance[row]), camera)
        ].append(int(row))
        temporal[camera] = int(frame)

    cameras_by_sequence_instance = defaultdict(list)
    for sequence, tree, camera in rows_by_sequence_instance_camera:
        cameras_by_sequence_instance[(sequence, tree)].append(camera)

    edges = []
    for key, camera_values in cameras_by_sequence_instance.items():
        camera_values = sorted(
            set(camera_values), key=lambda value: (temporal[value], value)
        )
        trees = {
            camera: cKDTree(
                xyz[
                    rows_by_sequence_instance_camera[
                        (key[0], key[1], camera)
                    ]
                ]
            )
            for camera in camera_values
        }
        for offset, first_camera in enumerate(camera_values):
            first_rows = np.asarray(
                rows_by_sequence_instance_camera[
                    (key[0], key[1], first_camera)
                ],
                dtype=np.int64,
            )
            for second_camera in camera_values[offset + 1 :]:
                if (
                    temporal[second_camera] - temporal[first_camera]
                    > int(maximum_frame_gap)
                ):
                    break
                second_rows = np.asarray(
                    rows_by_sequence_instance_camera[
                        (key[0], key[1], second_camera)
                    ],
                    dtype=np.int64,
                )
                distance_ab, nearest_ab = trees[second_camera].query(
                    xyz[first_rows], k=1, distance_upper_bound=float(radius)
                )
                distance_ba, nearest_ba = trees[first_camera].query(
                    xyz[second_rows], k=1, distance_upper_bound=float(radius)
                )
                valid = np.isfinite(distance_ab)
                first_local = np.flatnonzero(valid)
                if not len(first_local):
                    continue
                second_local = nearest_ab[valid].astype(np.int64)
                reciprocal = nearest_ba[second_local] == first_local
                for local_a, local_b, distance in zip(
                    first_local[reciprocal],
                    second_local[reciprocal],
                    distance_ab[valid][reciprocal],
                ):
                    if np.isfinite(distance_ba[local_b]):
                        edges.append(
                            (
                                float(distance),
                                int(first_rows[local_a]),
                                int(second_rows[local_b]),
                            )
                        )

    active_rows = np.flatnonzero(eligible)
    global_to_local = np.full(len(xyz), -1, dtype=np.int64)
    global_to_local[active_rows] = np.arange(len(active_rows))
    parent = np.arange(len(active_rows), dtype=np.int64)
    size = np.ones(len(active_rows), dtype=np.int16)
    camera_sets = [
        {int(cameras[row])} for row in active_rows
    ]
    frame_min = np.asarray(
        [
            camera_time.get(int(cameras[row]), ("", 0))[1]
            for row in active_rows
        ],
        dtype=np.int32,
    )
    frame_max = frame_min.copy()

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for _, global_a, global_b in sorted(edges):
        first = root(int(global_to_local[global_a]))
        second = root(int(global_to_local[global_b]))
        if first == second:
            continue
        if camera_sets[first] & camera_sets[second]:
            continue
        if int(size[first] + size[second]) > int(maximum_group_size):
            continue
        merged_min = min(int(frame_min[first]), int(frame_min[second]))
        merged_max = max(int(frame_max[first]), int(frame_max[second]))
        if merged_max - merged_min > int(maximum_frame_gap):
            continue
        if size[first] < size[second]:
            first, second = second, first
        parent[second] = first
        size[first] += size[second]
        camera_sets[first].update(camera_sets[second])
        frame_min[first] = merged_min
        frame_max[first] = merged_max

    components = defaultdict(list)
    for local, global_row in enumerate(active_rows):
        components[root(local)].append(int(global_row))
    return [
        np.asarray(rows, dtype=np.int64)
        for rows in components.values()
        if len(rows) >= 2
    ]


def _stable_unique(values) -> list[int]:
    result = []
    seen = set()
    for value in values:
        value = int(value)
        if value >= 0 and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _merge_sequence_groups(
    payload: dict,
    groups: list[np.ndarray],
    camera_time: dict[int, tuple[str, int]],
) -> tuple[dict, dict]:
    count = int(payload["centers"].shape[0])
    remove = np.zeros(count, dtype=bool)
    group_sizes = []
    for rows in groups:
        rows = np.asarray(rows, dtype=np.int64)
        points = payload["centers"][rows].numpy().astype(np.float64)
        distance = np.linalg.norm(
            points[:, None] - points[None], axis=2
        ).sum(axis=1)
        representative = int(rows[int(np.argmin(distance))])
        members = rows[rows != representative]
        remove[members] = True
        group_sizes.append(len(rows))

        covariance = payload["position_covariance"][rows].numpy()
        precision = 1.0 / np.maximum(
            np.trace(covariance, axis1=1, axis2=2), 1e-8
        )
        precision /= precision.sum()
        center = (points * precision[:, None]).sum(axis=0)
        centered = points - center[None]
        merged_covariance = (
            covariance * precision[:, None, None]
        ).sum(axis=0)
        if len(rows) > 1:
            merged_covariance += (
                centered.T @ (centered * precision[:, None])
            )
        payload["centers"][representative] = torch.from_numpy(
            center.astype(np.float32)
        )
        if "initialization_center" in payload:
            payload["initialization_center"][
                representative
            ] = torch.from_numpy(center.astype(np.float32))
        payload["position_covariance"][representative] = torch.from_numpy(
            merged_covariance.astype(np.float32)
        )
        payload["colors"][representative] = payload["colors"][
            rows
        ].mean(dim=0)
        payload["opacities"][representative] = payload["opacities"][
            rows
        ].median(dim=0).values
        payload["occupancy_probability"][
            representative
        ] = payload["occupancy_probability"][rows].mean()
        payload["reprojection_error"][representative] = torch.nanmean(
            payload["reprojection_error"][rows]
        )
        payload["ray_depth_nll"][representative] = torch.nanmean(
            payload["ray_depth_nll"][rows]
        )
        payload["free_space_violation_count"][
            representative
        ] = payload["free_space_violation_count"][rows].max()
        payload["unknown_view_count"][representative] = payload[
            "unknown_view_count"
        ][rows].max()
        payload["maximum_baseline"][representative] = payload[
            "maximum_baseline"
        ][rows].max()
        payload["maximum_triangulation_angle_degrees"][
            representative
        ] = payload["maximum_triangulation_angle_degrees"][rows].max()
        payload["track_id"][representative] = payload["track_id"][
            rows
        ].min()

        support = _stable_unique(
            payload["support_camera_ids"][rows].reshape(-1).tolist()
        )
        support.sort(
            key=lambda value: camera_time.get(value, ("", value))
        )
        support_width = int(payload["support_camera_ids"].shape[1])
        payload["support_camera_ids"][representative].fill_(-1)
        payload["support_camera_ids"][
            representative, : min(len(support), support_width)
        ] = torch.as_tensor(
            support[:support_width], dtype=torch.int32
        )
        payload["support_view_count"][representative] = len(support)
        payload["support_sequence_count"][representative] = len(
            {
                camera_time[value][0]
                for value in support
                if value in camera_time
            }
        )

        observations = []
        for row in rows:
            for camera, uv, depth in zip(
                payload["observation_camera_ids"][row].tolist(),
                payload["observation_uv"][row].tolist(),
                payload["observation_depth"][row].tolist(),
            ):
                if (
                    int(camera) >= 0
                    and np.isfinite(uv).all()
                    and np.isfinite(depth)
                ):
                    observations.append(
                        (int(camera), np.asarray(uv), float(depth))
                    )
        unique_observations = {}
        for observation in observations:
            unique_observations.setdefault(observation[0], observation)
        observations = sorted(
            unique_observations.values(),
            key=lambda value: camera_time.get(
                value[0], ("", value[0])
            ),
        )
        width = int(payload["observation_camera_ids"].shape[1])
        payload["observation_camera_ids"][representative].fill_(-1)
        payload["observation_uv"][representative].fill_(float("nan"))
        payload["observation_depth"][representative].fill_(float("nan"))
        for slot, (camera, uv, depth) in enumerate(
            observations[:width]
        ):
            payload["observation_camera_ids"][
                representative, slot
            ] = camera
            payload["observation_uv"][representative, slot] = (
                torch.from_numpy(uv.astype(np.float32))
            )
            payload["observation_depth"][representative, slot] = depth

    ray_offsets = payload.get("ray_evidence", {}).get("offsets")
    if ray_offsets is not None:
        bound_count = int(len(ray_offsets)) - 1
        if bool(remove[:bound_count].any()):
            raise RuntimeError(
                "Sequence-local merge attempted to remove a ray-bound "
                "canonical primitive"
            )
    keep = torch.from_numpy(~remove)
    for key, value in list(payload.items()):
        if torch.is_tensor(value) and value.ndim and len(value) == count:
            payload[key] = value[keep]
    audit = {
        "input_rows": count,
        "candidate_group_count": len(groups),
        "merged_input_rows": int(sum(group_sizes)),
        "removed_duplicate_rows": int(remove.sum()),
        "output_rows": int((~remove).sum()),
        "group_size_histogram": {
            str(size): int(group_sizes.count(size))
            for size in sorted(set(group_sizes))
        },
        "same_camera_rows_merged": 0,
        "ray_bound_rows_removed": 0,
    }
    return payload, audit


def repair_payload(
    payload: dict,
    *,
    skeleton_voxel_size: float,
    skeleton_minimum_confidence: float,
    sequence_merge_radius: float,
    sequence_merge_maximum_frame_gap: int,
    sequence_merge_maximum_group_size: int,
) -> tuple[dict, dict]:
    previous_repair = payload.get("audit", {}).get("contract_repair", {})
    structural_contract_exists = (
        "static_skeleton_confidence" in payload
        and bool(previous_repair.get("static_skeleton"))
        and bool(previous_repair.get("sequence_local_merge"))
    )
    if structural_contract_exists:
        # A v1 seed already contains the exact same skeleton and sequence
        # merge. Re-running reciprocal union-find can merge the retained
        # representatives a second time, so the migration is deliberately
        # idempotent and changes only the new optical-mass contract.
        skeleton_audit = dict(previous_repair["static_skeleton"])
        merge_audit = {
            "input_rows": int(len(payload["centers"])),
            "candidate_group_count": 0,
            "merged_input_rows": 0,
            "removed_duplicate_rows": 0,
            "output_rows": int(len(payload["centers"])),
            "group_size_histogram": {},
            "same_camera_rows_merged": 0,
            "ray_bound_rows_removed": 0,
            "reused_existing_contract": True,
            "source_protocol": previous_repair.get("protocol"),
        }
    else:
        layer = payload["layer_role"].numpy()
        canonical = layer == LAYER_CANONICAL_CROWN
        geometry = measured_static_skeleton_geometry(
            payload["centers"][canonical].numpy(),
            payload["colors"][canonical].numpy(),
            payload["tree_instance_id"][canonical].numpy(),
            payload["support_view_count"][canonical].numpy(),
            payload["support_sequence_count"][canonical].numpy(),
            payload["occupancy_probability"][canonical].numpy(),
            payload["ray_depth_nll"][canonical].numpy(),
            payload["free_space_violation_count"][canonical].numpy(),
            voxel_size=skeleton_voxel_size,
            minimum_confidence=skeleton_minimum_confidence,
        )
        canonical_rows = np.flatnonzero(canonical)
        skeleton_rows = canonical_rows[geometry["selected"]]
        payload["layer_role"][skeleton_rows] = LAYER_STATIC_SKELETON
        payload["scales"][skeleton_rows] = torch.from_numpy(
            geometry["scales"][geometry["selected"]]
        )
        payload["quaternions"][skeleton_rows] = torch.from_numpy(
            geometry["quaternions"][geometry["selected"]]
        )
        payload["track_linearity"][canonical_rows] = torch.maximum(
            payload["track_linearity"][canonical_rows],
            torch.from_numpy(geometry["linearity"]),
        )
        skeleton_confidence = torch.zeros(
            len(layer), dtype=torch.float32
        )
        skeleton_confidence[canonical_rows] = torch.from_numpy(
            geometry["confidence"]
        )
        payload["static_skeleton_confidence"] = skeleton_confidence
        skeleton_audit = {
            "canonical_candidates": int(canonical.sum()),
            "selected": int(len(skeleton_rows)),
            "minimum_confidence": float(
                skeleton_minimum_confidence
            ),
            "confidence_quantiles": np.quantile(
                geometry["confidence"],
                [0.5, 0.9, 0.99, 0.999, 1.0],
            ).tolist(),
            "linearity_quantiles": np.quantile(
                geometry["linearity"],
                [0.5, 0.9, 0.99, 0.999, 1.0],
            ).tolist(),
            "evidence_contract": (
                "cross_sequence_local_line_cylinder_continuous_ray_rgb"
            ),
        }

        camera_time = _camera_time_lookup(payload.get("audit", {}))
        primary_camera = _primary_observation_camera(payload)
        dynamic_dense = (
            (payload["layer_role"].numpy() == LAYER_DYNAMIC_LEAF)
            & (
                payload["initialization_source"].numpy()
                == SOURCE_DENSE_RAY
            )
            & (primary_camera >= 0)
        )
        groups = sequence_local_correspondence_groups(
            payload["centers"].numpy(),
            payload["tree_instance_id"].numpy(),
            primary_camera,
            camera_time,
            dynamic_dense,
            radius=sequence_merge_radius,
            maximum_frame_gap=sequence_merge_maximum_frame_gap,
            maximum_group_size=sequence_merge_maximum_group_size,
        )
        payload, merge_audit = _merge_sequence_groups(
            payload, groups, camera_time
        )
    dense_dynamic = (
        (payload["layer_role"] == LAYER_DYNAMIC_LEAF)
        & (payload["initialization_source"] == SOURCE_DENSE_RAY)
    )
    initial_opacity, training_floor = (
        evidence_conditioned_leaf_optical_mass(
            payload["occupancy_probability"],
            payload["support_view_count"],
            payload["unknown_view_count"],
            payload["ray_depth_nll"],
            payload["free_space_violation_count"],
        )
    )
    previous_opacity = payload["opacities"].reshape(-1)
    matured_opacity = torch.where(
        dense_dynamic,
        torch.maximum(previous_opacity, initial_opacity),
        previous_opacity,
    )
    payload["opacities"] = matured_opacity[:, None]
    dense_initial = matured_opacity[dense_dynamic]
    dense_floor = training_floor[dense_dynamic]
    audit = {
        "protocol": PROTOCOL,
        "static_skeleton": skeleton_audit,
        "sequence_local_merge": {
            **merge_audit,
            "radius_m": float(sequence_merge_radius),
            "maximum_frame_gap": int(
                sequence_merge_maximum_frame_gap
            ),
            "maximum_group_size": int(
                sequence_merge_maximum_group_size
            ),
            "correspondence_contract": (
                "reciprocal_nearest_cross_camera_same_sequence_instance"
            ),
        },
        "dynamic_optical_mass": {
            "candidate_count": int(dense_dynamic.sum()),
            "initial_opacity_quantiles": torch.quantile(
                dense_initial,
                torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]),
            ).tolist(),
            "training_floor_quantiles": torch.quantile(
                dense_floor,
                torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]),
            ).tolist(),
            "contract": (
                "per_candidate_occupancy_support_depth_free_space_posterior"
            ),
            "historical_model_or_rgb_fit_used": False,
        },
    }
    payload["geometry_version"] = (
        str(payload.get("geometry_version", "unknown"))
        + "_measured_skeleton_sequence_graph_optical_mass_v2"
    )
    payload.setdefault("audit", {})["contract_repair"] = audit
    payload["audit"]["static_skeleton"] = int(
        (payload["layer_role"] == LAYER_STATIC_SKELETON).sum()
    )
    payload["audit"]["sequence_local_dynamic_tracks"] = int(
        (payload["layer_role"] == LAYER_DYNAMIC_LEAF).sum()
    )
    return payload, audit


def main() -> None:
    args = _parse_args()
    source = args.source_initialization.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    required = (
        "initialization_manifest.json",
        "foliage_seed_gaussians.pth",
        "foliage_seed_gaussians.json",
        "surface_seed.npz",
        "surface_seed.json",
    )
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    payload = torch.load(
        source / "foliage_seed_gaussians.pth", map_location="cpu"
    )
    payload, audit = repair_payload(
        payload,
        skeleton_voxel_size=args.skeleton_voxel_size,
        skeleton_minimum_confidence=(
            args.skeleton_minimum_confidence
        ),
        sequence_merge_radius=args.sequence_merge_radius,
        sequence_merge_maximum_frame_gap=(
            args.sequence_merge_maximum_frame_gap
        ),
        sequence_merge_maximum_group_size=(
            args.sequence_merge_maximum_group_size
        ),
    )

    output.mkdir(parents=True)
    for name in ("surface_seed.npz", "surface_seed.json"):
        shutil.copy2(source / name, output / name)
    foliage_path = output / "foliage_seed_gaussians.pth"
    torch.save(payload, foliage_path)

    summary = json.loads(
        (source / "foliage_seed_gaussians.json").read_text(
            encoding="utf-8"
        )
    )
    summary["contract_repair"] = audit
    summary["static_skeleton"] = int(
        (payload["layer_role"] == LAYER_STATIC_SKELETON).sum()
    )
    summary["sequence_local_dynamic_tracks"] = int(
        (payload["layer_role"] == LAYER_DYNAMIC_LEAF).sum()
    )
    summary["foliage_count"] = int(len(payload["centers"]))
    summary["migrated_from"] = {
        "initialization": str(source),
        "foliage_seed_sha256": sha256_file(
            source / "foliage_seed_gaussians.pth"
        ),
        "migration": PROTOCOL,
    }
    (output / "foliage_seed_gaussians.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    manifest = json.loads(
        (source / "initialization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["surface_seed"] = str(output / "surface_seed.npz")
    manifest["foliage_seed"] = str(foliage_path)
    manifest["foliage"] = summary
    manifest.setdefault("initialization_contract", {})[
        "foliage_physical_ownership"
    ] = PROTOCOL
    manifest["migrated_from"] = {
        "initialization": str(source),
        "manifest_sha256": sha256_file(
            source / "initialization_manifest.json"
        ),
        "foliage_seed_sha256": sha256_file(
            source / "foliage_seed_gaussians.pth"
        ),
        "render_and_training_equivalent": False,
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "foliage_seed_sha256": sha256_file(foliage_path),
                **audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
