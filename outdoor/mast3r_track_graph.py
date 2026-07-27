"""Build fixed-camera multi-view tracks from aligned MASt3R pointmaps.

This module deliberately never reads ``points3D.bin``.  MASt3R supplies dense
per-pixel geometry and confidence, while the calibrated pointmap cameras
provide rays.  Tracks are accepted only after independent target-pointmap
agreement and fixed-camera triangulation.  The resulting archive therefore
has real, ragged pixel observations instead of the one-camera sparse export
previously called ``mast3r_tracks``.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback remains supported.
    orjson = None

from matcha.cambridge_masks import CambridgeMaskLookup


TRACK_GRAPH_VERSION = "mast3r-fixed-camera-multiview-track-graph-v2-exact-k"
ROLE_NAMES = ("rigid", "canopy", "sky", "transient", "unknown")
ROLE_RIGID, ROLE_CANOPY, ROLE_SKY, ROLE_TRANSIENT, ROLE_UNKNOWN = range(5)


def _sequence(name: str) -> str:
    stem = Path(name).stem
    return stem.split("__", 1)[0] if "__" in stem else "default"


def _frame_number(name: str) -> int:
    match = re.search(r"frame(\d+)", Path(name).stem)
    return int(match.group(1)) if match else 0


class _PointmapCache:
    def __init__(self, root: Path, *, capacity: int = 10):
        self.root = Path(root)
        self.capacity = max(int(capacity), 2)
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, stem: str) -> dict[str, np.ndarray]:
        value = self.values.pop(stem, None)
        if value is not None:
            self.values[stem] = value
            return value
        path = self.root / f"{stem}.json"
        raw = path.read_bytes()
        payload = orjson.loads(raw) if orjson is not None else json.loads(raw)
        confidence = np.asarray(payload["confs"], dtype=np.float32)
        height, width = confidence.shape
        points = np.asarray(payload["points"], dtype=np.float32).reshape(
            height, width, 3
        )
        rgb = payload.get("rgb")
        if rgb is None:
            colors = np.zeros((height, width, 3), dtype=np.float32)
        else:
            colors = np.asarray(rgb, dtype=np.float32).reshape(
                height, width, 3
            )
        value = {"points": points, "confidence": confidence, "rgb": colors}
        self.values[stem] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def _camera_arrays(
    scene: Path,
    *,
    dataset: Path | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load per-view ``fx, fy, cx, cy`` without inventing a centred camera.

    New MASt3R runs persist ``intrinsics`` directly.  Older chart archives can
    only be upgraded from the immutable Cambridge camera database; their
    scalar focal field is deliberately not accepted as an exact-K contract.
    """
    payload = json.loads((Path(scene) / "cameras.json").read_text())
    paths = payload.get("filepaths")
    c2w = np.asarray(payload.get("cams2world"), dtype=np.float64)
    if not isinstance(paths, list) or c2w.shape != (len(paths), 4, 4):
        raise RuntimeError("Unsupported MASt3R pointmap camera contract")
    names = [Path(value).stem for value in paths]
    if len(set(names)) != len(names):
        raise RuntimeError("Pointmap camera stems are not unique")
    stored = payload.get("intrinsics")
    if stored is not None:
        intrinsics = np.asarray(stored, dtype=np.float64)
        if intrinsics.shape != (len(names), 4):
            raise RuntimeError("MASt3R intrinsics must have shape [N,4]")
        return names, intrinsics, c2w
    if dataset is None or image_size is None:
        raise RuntimeError(
            "Legacy scalar-focal cameras.json is not an exact-K contract. "
            "Pass the calibrated Cambridge dataset or regenerate MASt3R."
        )
    # Import lazily so synthetic/unit users of this module do not require the
    # COLMAP reader. Reading cameras/images is allowed; points3D is never read.
    from outdoor.scene_contract import _camera_intrinsics
    from colmap.read_write_model import read_cameras_binary, read_images_binary

    sparse = Path(dataset) / "sparse" / "0"
    cameras = read_cameras_binary(str(sparse / "cameras.bin"))
    images = read_images_binary(str(sparse / "images.bin"))
    by_stem = {Path(image.name).stem: image for image in images.values()}
    target_width, target_height = map(int, image_size)
    rows = []
    for name in names:
        image = by_stem.get(name)
        if image is None:
            raise RuntimeError(f"Exact-K camera missing for Chart view {name}")
        camera = _camera_intrinsics(cameras[image.camera_id])
        if camera["fx"] is None or camera["distortion"]:
            raise RuntimeError(
                f"Track graph requires an undistorted pinhole camera: {name}"
            )
        sx = target_width / float(camera["width"])
        sy = target_height / float(camera["height"])
        rows.append(
            [
                float(camera["fx"]) * sx,
                float(camera["fy"]) * sy,
                float(camera["cx"]) * sx,
                float(camera["cy"]) * sy,
            ]
        )
    return names, np.asarray(rows, dtype=np.float64), c2w


def _project(
    xyz: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray | float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    values = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if len(values) == 1:  # Backwards-compatible synthetic-test shorthand.
        fx = fy = float(values[0])
        cx, cy = width * 0.5, height * 0.5
    elif len(values) == 4:
        fx, fy, cx, cy = map(float, values)
    else:
        raise ValueError("intrinsics must be scalar or [fx,fy,cx,cy]")
    u = fx * camera[:, 0] / np.maximum(z, 1e-8) + cx
    v = fy * camera[:, 1] / np.maximum(z, 1e-8) + cy
    valid = (
        np.isfinite(camera).all(axis=1)
        & (z > 0.05)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return u, v, z, valid


def _pointmap_overlap(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    *,
    stride: int,
    confidence_threshold: float,
    absolute_gate: float,
    relative_gate: float,
) -> float:
    height, width = source["confidence"].shape
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    xyz = source["points"][rows, columns].reshape(-1, 3)
    source_conf = source["confidence"][rows, columns].reshape(-1)
    u, v, depth, valid = _project(xyz, c2w, intrinsics, width, height)
    target_rows = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
    target_columns = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
    target_xyz = target["points"][target_rows, target_columns]
    target_conf = target["confidence"][target_rows, target_columns]
    distance = np.linalg.norm(target_xyz - xyz, axis=1)
    gate = np.maximum(float(absolute_gate), float(relative_gate) * depth)
    eligible = (
        valid
        & (source_conf >= confidence_threshold)
        & (target_conf >= confidence_threshold)
    )
    denominator = max(int(eligible.sum()), 1)
    return float((eligible & (distance <= gate)).sum() / denominator)


def build_pair_graph(
    scene: Path,
    *,
    dataset: Path | None = None,
    temporal_neighbours: int = 4,
    cross_sequence_neighbours: int = 8,
    candidate_neighbours: int = 20,
    overlap_stride: int = 16,
    minimum_overlap: float = 0.03,
    confidence_threshold: float = 1.25,
    absolute_gate: float = 0.10,
    relative_gate: float = 0.012,
    cache_size: int = 64,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[int, list[int]], dict[str, Any]]:
    """Build temporal, retrieval-proxy and measured-overlap pair edges."""
    scene = Path(scene)
    camera_payload = json.loads((scene / "cameras.json").read_text())
    first_name = Path(camera_payload["filepaths"][0]).stem
    first_map = _PointmapCache(scene / "pointmaps", capacity=2).get(first_name)
    pointmap_size = (
        int(first_map["confidence"].shape[1]),
        int(first_map["confidence"].shape[0]),
    )
    names, intrinsics, c2w = _camera_arrays(
        scene, dataset=dataset, image_size=pointmap_size
    )
    centers = c2w[:, :3, 3]
    forward = c2w[:, :3, 2]
    extent = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-6)
    features = np.concatenate(
        [(centers - centers.mean(0)) / extent, 0.35 * forward], axis=1
    )
    sequences = [_sequence(name) for name in names]
    cache = _PointmapCache(scene / "pointmaps", capacity=cache_size)
    graph: dict[int, list[int]] = {}
    edge_overlap: dict[str, float] = {}
    for source, name in enumerate(names):
        same = [
            index
            for index in range(len(names))
            if index != source and sequences[index] == sequences[source]
        ]
        same.sort(
            key=lambda index: (
                abs(_frame_number(names[index]) - _frame_number(name)),
                index,
            )
        )
        temporal = same[: int(temporal_neighbours)]
        other = np.asarray(
            [
                index
                for index in range(len(names))
                if index != source and sequences[index] != sequences[source]
            ],
            dtype=np.int64,
        )
        if len(other):
            distance = np.linalg.norm(
                features[other] - features[source][None], axis=1
            )
            candidates = other[np.argsort(distance, kind="stable")][
                : int(candidate_neighbours)
            ]
        else:
            candidates = np.empty(0, dtype=np.int64)
        source_map = cache.get(name)
        ranked: list[tuple[float, int]] = []
        for target in candidates:
            overlap = _pointmap_overlap(
                source_map,
                cache.get(names[int(target)]),
                c2w[int(target)],
                intrinsics[int(target)],
                stride=overlap_stride,
                confidence_threshold=confidence_threshold,
                absolute_gate=absolute_gate,
                relative_gate=relative_gate,
            )
            edge_overlap[f"{source}:{int(target)}"] = overlap
            if overlap >= float(minimum_overlap):
                ranked.append((overlap, int(target)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        cross = [target for _, target in ranked[: int(cross_sequence_neighbours)]]
        graph[source] = sorted(set(temporal + cross))
    audit = {
        "version": TRACK_GRAPH_VERSION,
        "view_count": len(names),
        "edge_count_directed": int(sum(map(len, graph.values()))),
        "temporal_neighbours": int(temporal_neighbours),
        "cross_sequence_neighbours": int(cross_sequence_neighbours),
        "minimum_overlap": float(minimum_overlap),
        "edge_overlap": edge_overlap,
    }
    audit["intrinsics_policy"] = "per_view_exact_fx_fy_cx_cy"
    return names, intrinsics, c2w, graph, audit


def _triangulate(
    pixels: np.ndarray,
    camera_indices: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    centers = c2w[camera_indices, :3, 3]
    camera_k = np.asarray(intrinsics)[camera_indices]
    if camera_k.ndim == 1:
        fx = fy = camera_k
        cx = np.full(len(pixels), width * 0.5)
        cy = np.full(len(pixels), height * 0.5)
    else:
        fx, fy, cx, cy = camera_k.T
    local = np.column_stack(
        [
            (pixels[:, 0] - cx) / fx,
            (pixels[:, 1] - cy) / fy,
            np.ones(len(pixels)),
        ]
    )
    directions = np.einsum(
        "nij,nj->ni", c2w[camera_indices, :3, :3], local
    )
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-9
    )
    identity = np.eye(3, dtype=np.float64)
    projectors = identity[None] - directions[:, :, None] * directions[:, None, :]
    weighted = projectors * weights[:, None, None]
    information = weighted.sum(axis=0)
    rhs = np.einsum("nij,nj->i", weighted, centers)
    try:
        xyz = np.linalg.solve(information + identity * 1e-8, rhs)
    except np.linalg.LinAlgError:
        xyz = np.linalg.lstsq(information + identity * 1e-8, rhs, rcond=None)[0]
    reprojection = []
    for pixel, index in zip(pixels, camera_indices):
        u, v, _, valid = _project(
            xyz[None], c2w[int(index)], intrinsics[int(index)], width, height
        )
        reprojection.append(
            float(np.linalg.norm(np.asarray([u[0], v[0]]) - pixel))
            if valid[0]
            else 1e6
        )
    dots = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(dots[np.triu_indices(len(dots), 1)])))
    angle_stats = (
        np.percentile(angles, [10, 50, 90]).astype(np.float32)
        if len(angles)
        else np.zeros(3, dtype=np.float32)
    )
    residual = max(float(np.median(reprojection)), 0.25)
    angular_sigma = residual / max(
        float(np.median(np.sqrt(fx * fy))), 1.0
    )
    covariance = (
        np.linalg.pinv(information + identity * 1e-8)
        * angular_sigma**2
    )
    return (
        xyz.astype(np.float32),
        np.asarray(reprojection, dtype=np.float32),
        float(np.median(reprojection)),
        np.concatenate([angle_stats, np.diag(covariance).astype(np.float32)]),
    )


def _mask_role_maps(
    lookup: CambridgeMaskLookup | None,
    names: list[str],
) -> list[np.ndarray | None]:
    result: list[np.ndarray | None] = []
    for name in names:
        if lookup is None:
            result.append(None)
            continue
        key = lookup.source_name_for(name)
        channels = lookup.masks[key]
        values = np.stack(
            [
                channel.detach().cpu().bool().numpy()
                for channel in channels[:4]
            ],
            axis=0,
        )
        result.append(values)
    return result


def _roles_at(
    maps: np.ndarray | None,
    pixels: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if maps is None:
        return np.full(len(pixels), ROLE_UNKNOWN, dtype=np.int8)
    rows = np.clip(
        np.floor(pixels[:, 1] / height * maps.shape[1]).astype(np.int64),
        0,
        maps.shape[1] - 1,
    )
    columns = np.clip(
        np.floor(pixels[:, 0] / width * maps.shape[2]).astype(np.int64),
        0,
        maps.shape[2] - 1,
    )
    object_keep, sky_keep, distortion_keep, tree_keep = maps[:, rows, columns]
    role = np.full(len(pixels), ROLE_RIGID, dtype=np.int8)
    role[~tree_keep] = ROLE_CANOPY
    role[~sky_keep] = ROLE_SKY
    role[~object_keep | ~distortion_keep] = ROLE_TRANSIENT
    return role


def build_multiview_tracks(
    scene: Path,
    output: Path,
    *,
    dataset: Path | None = None,
    tree_mask_pickle: Path | None = None,
    stride: int = 5,
    confidence_threshold: float = 1.25,
    absolute_gate: float = 0.10,
    relative_gate: float = 0.012,
    minimum_observations: int = 2,
    maximum_reprojection_error: float = 2.0,
    minimum_triangulation_angle: float = 0.35,
    dedup_voxel: float = 0.025,
    cache_size: int = 64,
    pair_graph_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct and persist auditable global and sequence-local tracks."""
    scene = Path(scene).resolve()
    output = Path(output).resolve()
    names, intrinsics, c2w, graph, graph_audit = build_pair_graph(
        scene,
        dataset=dataset,
        cache_size=cache_size,
        **(pair_graph_kwargs or {}),
    )
    cache = _PointmapCache(scene / "pointmaps", capacity=cache_size)
    first = cache.get(names[0])
    height, width = first["confidence"].shape
    lookup = (
        CambridgeMaskLookup(
            Path(dataset), Path(tree_mask_pickle), mask_indices=[0, 1, 2, 3]
        )
        if dataset is not None and tree_mask_pickle is not None
        else None
    )
    role_maps = _mask_role_maps(lookup, names)
    sequences = [_sequence(name) for name in names]
    records: list[dict[str, Any]] = []
    occupied_track_cells: set[tuple[int, int, int]] = set()
    for source, name in enumerate(names):
        source_map = cache.get(name)
        rows, columns = np.mgrid[0:height:stride, 0:width:stride]
        pixels = np.column_stack([columns.reshape(-1), rows.reshape(-1)]).astype(
            np.float32
        )
        xyz = source_map["points"][rows, columns].reshape(-1, 3)
        confidence = source_map["confidence"][rows, columns].reshape(-1)
        colors = source_map["rgb"][rows, columns].reshape(-1, 3)
        valid_source = (
            np.isfinite(xyz).all(axis=1)
            & (np.linalg.norm(xyz, axis=1) > 1e-5)
            & (confidence >= float(confidence_threshold))
        )
        candidate_indices = np.flatnonzero(valid_source)
        source_roles = _roles_at(
            role_maps[source], pixels, width, height
        )
        observations: list[list[tuple[int, float, float, float, int]]] = [
            [
                (
                    source,
                    float(pixels[index, 0]),
                    float(pixels[index, 1]),
                    float(confidence[index]),
                    int(source_roles[index]),
                )
            ]
            for index in candidate_indices
        ]
        point_values: list[list[np.ndarray]] = [
            [xyz[index].astype(np.float32)] for index in candidate_indices
        ]
        for target in graph[source]:
            target_map = cache.get(names[target])
            anchor_xyz = xyz[candidate_indices]
            u, v, depth, projected = _project(
                anchor_xyz, c2w[target], intrinsics[target], width, height
            )
            tr = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
            tc = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
            target_xyz = target_map["points"][tr, tc]
            target_conf = target_map["confidence"][tr, tc]
            distance = np.linalg.norm(target_xyz - anchor_xyz, axis=1)
            gate = np.maximum(float(absolute_gate), float(relative_gate) * depth)
            accepted = (
                projected
                & np.isfinite(target_xyz).all(axis=1)
                & (target_conf >= float(confidence_threshold))
                & (distance <= gate)
            )
            target_pixels = np.column_stack([u, v]).astype(np.float32)
            target_roles = _roles_at(
                role_maps[target], target_pixels, width, height
            )
            for local in np.flatnonzero(accepted):
                observations[local].append(
                    (
                        target,
                        float(u[local]),
                        float(v[local]),
                        float(target_conf[local]),
                        int(target_roles[local]),
                    )
                )
                point_values[local].append(target_xyz[local])
        for local, obs in enumerate(observations):
            unique = {}
            for item in obs:
                existing = unique.get(item[0])
                if existing is None or item[3] > existing[3]:
                    unique[item[0]] = item
            obs = list(unique.values())
            if len(obs) < int(minimum_observations):
                continue
            # Pointmaps from nearby charts describe the same dense surface.
            # Reject already-owned spatial cells before the comparatively
            # expensive fixed-ray solve; the former post-hoc deduplication
            # spent most runtime triangulating tracks that could never survive.
            candidate_cell = tuple(
                np.floor(
                    xyz[candidate_indices[local]] / float(dedup_voxel)
                ).astype(int)
            )
            if candidate_cell in occupied_track_cells:
                continue
            camera_indices = np.asarray([item[0] for item in obs], dtype=np.int32)
            obs_pixels = np.asarray(
                [[item[1], item[2]] for item in obs], dtype=np.float32
            )
            obs_conf = np.asarray([item[3] for item in obs], dtype=np.float32)
            tri_xyz, residuals, median_error, tri = _triangulate(
                obs_pixels,
                camera_indices,
                c2w,
                intrinsics,
                width,
                height,
                np.clip(obs_conf / np.median(obs_conf), 0.25, 4.0),
            )
            if (
                median_error > float(maximum_reprojection_error)
                or tri[1] < float(minimum_triangulation_angle)
            ):
                continue
            occupied_track_cells.add(candidate_cell)
            roles = np.asarray([item[4] for item in obs], dtype=np.int8)
            role_count = np.bincount(roles, minlength=len(ROLE_NAMES)).astype(
                np.int16
            )
            support_sequences = len({sequences[int(index)] for index in camera_indices})
            agreement = np.asarray(point_values[local], dtype=np.float32)
            spread = float(
                np.median(np.linalg.norm(agreement - np.median(agreement, 0), axis=1))
            )
            covariance_diag = np.maximum(
                tri[3:],
                (spread / math.sqrt(len(obs))) ** 2,
            )
            records.append(
                {
                    "xyz": tri_xyz,
                    "rgb": np.clip(colors[candidate_indices[local]] * 255, 0, 255).astype(
                        np.uint8
                    ),
                    "camera_indices": camera_indices,
                    "pixels": obs_pixels,
                    "confidence": obs_conf,
                    "residuals": residuals,
                    "camera_depths": np.asarray(
                        [
                            _project(
                                tri_xyz[None],
                                c2w[int(camera_index)],
                                intrinsics[int(camera_index)],
                                width,
                                height,
                            )[2][0]
                            for camera_index in camera_indices
                        ],
                        dtype=np.float32,
                    ),
                    "centered_residuals": np.asarray(
                        [
                            np.linalg.norm(
                                np.asarray(
                                    _project(
                                        tri_xyz[None],
                                        c2w[int(camera_index)],
                                        float(
                                            np.sqrt(
                                                intrinsics[int(camera_index), 0]
                                                * intrinsics[int(camera_index), 1]
                                            )
                                        ),
                                        width,
                                        height,
                                    )[:2]
                                ).reshape(2)
                                - pixel
                            )
                            for pixel, camera_index in zip(
                                obs_pixels, camera_indices
                            )
                        ],
                        dtype=np.float32,
                    ),
                    "median_error": median_error,
                    "angle": tri[:3],
                    "covariance_diag": np.maximum(
                        covariance_diag, 1e-8
                    ),
                    "role_count": role_count,
                    "sequence_count": support_sequences,
                    "cycle_consistency": math.exp(-spread / max(absolute_gate, 1e-6)),
                    "pointmap_spread": spread,
                }
            )
    if not records:
        raise RuntimeError("No fixed-camera MASt3R multi-view tracks survived")
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        [len(record["camera_indices"]) for record in records], dtype=np.int64
    )
    role_counts = np.stack([record["role_count"] for record in records])
    role_probabilities = role_counts / np.maximum(
        role_counts.sum(axis=1, keepdims=True), 1
    )
    arrays = {
        "schema_version": np.asarray(TRACK_GRAPH_VERSION),
        "track_id": np.arange(len(records), dtype=np.int64),
        "xyz": np.stack([record["xyz"] for record in records]).astype(np.float32),
        "rgb": np.stack([record["rgb"] for record in records]).astype(np.uint8),
        "reprojection_error": np.asarray(
            [record["median_error"] for record in records], dtype=np.float32
        ),
        "valid_observation_count": np.asarray(
            [len(record["camera_indices"]) for record in records], dtype=np.int16
        ),
        "sequence_count": np.asarray(
            [record["sequence_count"] for record in records], dtype=np.int16
        ),
        "role_observation_count": role_counts.astype(np.int16),
        "role_probabilities": role_probabilities.astype(np.float32),
        "dominant_role": np.argmax(role_counts, axis=1).astype(np.int8),
        "position_covariance_diag": np.stack(
            [record["covariance_diag"] for record in records]
        ).astype(np.float32),
        "triangulation_angle_p10": np.asarray(
            [record["angle"][0] for record in records], dtype=np.float32
        ),
        "triangulation_angle_median": np.asarray(
            [record["angle"][1] for record in records], dtype=np.float32
        ),
        "triangulation_angle_p90": np.asarray(
            [record["angle"][2] for record in records], dtype=np.float32
        ),
        "cycle_consistency": np.asarray(
            [record["cycle_consistency"] for record in records], dtype=np.float32
        ),
        "pointmap_spread": np.asarray(
            [record["pointmap_spread"] for record in records], dtype=np.float32
        ),
        "observation_offsets": offsets,
        "observation_camera_indices": np.concatenate(
            [record["camera_indices"] for record in records]
        ).astype(np.int32),
        "observation_pixels": np.concatenate(
            [record["pixels"] for record in records]
        ).astype(np.float32),
        "observation_confidence": np.concatenate(
            [record["confidence"] for record in records]
        ).astype(np.float32),
        "observation_reprojection_error": np.concatenate(
            [record["residuals"] for record in records]
        ).astype(np.float32),
        "observation_camera_depth": np.concatenate(
            [record["camera_depths"] for record in records]
        ).astype(np.float32),
        "observation_centered_reprojection_error": np.concatenate(
            [record["centered_residuals"] for record in records]
        ).astype(np.float32),
        "camera_names": np.asarray(names),
        "camera_image_sizes": np.tile(
            np.asarray([[width, height]], dtype=np.int32), (len(names), 1)
        ),
        # Compatibility aliases consumed by the current teacher.
        "support_camera_offsets": offsets,
        "support_camera_ids": np.concatenate(
            [record["camera_indices"] for record in records]
        ).astype(np.int32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    observations = arrays["valid_observation_count"]
    sequences_array = arrays["sequence_count"]
    rigid = arrays["dominant_role"] == ROLE_RIGID
    canopy = arrays["dominant_role"] == ROLE_CANOPY
    summary = {
        **graph_audit,
        "schema_version": TRACK_GRAPH_VERSION,
        "source": "aligned_mast3r_pointmaps_and_fixed_camera_rays",
        "points3D_bin_read": False,
        "colmap_tracks_read": False,
        "track_count": len(records),
        "observation_count": int(offsets[-1]),
        "observation_count_percentiles": np.percentile(
            observations, [0, 10, 50, 90, 100]
        ).tolist(),
        "sequence_count_percentiles": np.percentile(
            sequences_array, [0, 10, 50, 90, 100]
        ).tolist(),
        "reprojection_error_percentiles": np.percentile(
            arrays["reprojection_error"], [10, 50, 90, 99]
        ).tolist(),
        "exact_k_observation_reprojection_percentiles": np.percentile(
            arrays["observation_reprojection_error"], [50, 90, 99]
        ).tolist(),
        "centered_k_observation_reprojection_percentiles": np.percentile(
            arrays["observation_centered_reprojection_error"], [50, 90, 99]
        ).tolist(),
        "triangulation_angle_median_percentiles": np.percentile(
            arrays["triangulation_angle_median"], [10, 50, 90]
        ).tolist(),
        "rigid_track_count": int(rigid.sum()),
        "global_rigid_track_count": int((rigid & (sequences_array >= 2)).sum()),
        "sequence_local_foliage_track_count": int(
            (canopy & (observations >= 2)).sum()
        ),
        "global_foliage_anchor_count": int(
            (canopy & (sequences_array >= 2)).sum()
        ),
        "camera_scope": "selected_matcha_chart_views",
        "full_database_camera_count": None,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def validate_track_gate(
    archive_path: Path,
    *,
    minimum_tracks: int = 10_000,
    minimum_global_rigid_tracks: int = 2_000,
    minimum_median_observations: float = 2.0,
    minimum_cross_sequence_fraction: float = 0.05,
    maximum_median_reprojection_error: float = 1.5,
    minimum_median_triangulation_angle: float = 0.5,
) -> dict[str, Any]:
    with np.load(archive_path, allow_pickle=False) as archive:
        count = len(archive["track_id"])
        observations = archive["valid_observation_count"]
        sequences = archive["sequence_count"]
        rigid = archive["dominant_role"] == ROLE_RIGID
        reprojection = archive["reprojection_error"]
        angle = archive["triangulation_angle_median"]
    values = {
        "track_count": int(count),
        "global_rigid_track_count": int((rigid & (sequences >= 2)).sum()),
        "median_observations": float(np.median(observations)),
        "cross_sequence_fraction": float((sequences >= 2).mean()),
        "median_reprojection_error": float(np.median(reprojection)),
        "median_triangulation_angle": float(np.median(angle)),
    }
    failures = []
    checks = {
        "track_count": values["track_count"] >= minimum_tracks,
        "global_rigid_tracks": values["global_rigid_track_count"]
        >= minimum_global_rigid_tracks,
        "median_observations": values["median_observations"]
        >= minimum_median_observations,
        "cross_sequence_fraction": values["cross_sequence_fraction"]
        >= minimum_cross_sequence_fraction,
        "median_reprojection_error": values["median_reprojection_error"]
        <= maximum_median_reprojection_error,
        "median_triangulation_angle": values["median_triangulation_angle"]
        >= minimum_median_triangulation_angle,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    result = {"passed": not failures, "checks": checks, "values": values, "failures": failures}
    if failures:
        raise RuntimeError(
            "MASt3R multi-view track gate failed: " + ", ".join(failures)
        )
    return result
