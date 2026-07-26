"""Role-aware initialization from a unified outdoor evidence store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.evidence_store import (
    ROLE_CANOPY,
    ROLE_RIGID,
    ROLE_SKY,
    ROLE_TRANSIENT,
    artifact_path,
    load_evidence_store,
)
from outdoor.foliage_geometry import (
    _camera_record,
    build_instance_aware_canopy_volume,
    cluster_tree_instances,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary_with_tracks,
    semantic_tree_tracks,
)
from outdoor.foliage_view_graph import greedy_diverse_views
from scripts.augment_foliage_seed_with_sfm_tracks import _track_frames


INITIALIZATION_VERSION = "outdoor-role-aware-initialization-v1"

SOURCE_COLMAP = 0
SOURCE_MAST3R = 1
SOURCE_CHART = 2


def _load_tracks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _selection_mask(
    tracks: dict[str, np.ndarray],
    *,
    minimum_rigid_probability: float,
    maximum_canopy_probability: float,
    minimum_observations: int,
    maximum_reprojection_error: float,
) -> np.ndarray:
    probability = tracks["role_probabilities"]
    return (
        (probability[:, ROLE_RIGID] >= float(minimum_rigid_probability))
        & (probability[:, ROLE_CANOPY] <= float(maximum_canopy_probability))
        & (probability[:, ROLE_SKY] <= 0.05)
        & (probability[:, ROLE_TRANSIENT] <= 0.05)
        & (
            tracks["valid_observation_count"]
            >= int(minimum_observations)
        )
        & (
            tracks["reprojection_error"]
            <= float(maximum_reprojection_error)
        )
    )


def _deduplicate(
    candidates: np.ndarray,
    reference: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    if not len(candidates):
        return np.zeros(0, dtype=bool)
    if not len(reference):
        return np.ones(len(candidates), dtype=bool)
    distance = cKDTree(reference).query(candidates, k=1)[0]
    return distance > float(radius)


def _surface_frames(
    xyz: np.ndarray,
    position_sigma: np.ndarray,
    *,
    minimum_scale: float = 0.004,
    maximum_scale: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate tangent scales, orientation and normals from local support."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 3:
        raise RuntimeError("At least three rigid seeds are required")
    neighbors = min(12, len(xyz))
    distances, indices = cKDTree(xyz).query(xyz, k=neighbors)
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    local = xyz[indices] - xyz[:, None]
    covariance = np.einsum("nki,nkj->nij", local, local) / max(
        neighbors - 1, 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, order, axis=1)
    eigenvectors = np.take_along_axis(
        eigenvectors, order[:, None, :], axis=2
    )
    frames = np.transpose(eigenvectors, (0, 2, 1))
    negative = np.linalg.det(frames) < 0
    frames[negative, 2] *= -1
    xyzw = Rotation.from_matrix(frames).as_quat().astype(np.float32)
    quaternion = np.column_stack(
        [xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]]
    ).astype(np.float32)
    nearest = (
        distances[:, 1]
        if distances.shape[1] > 1
        else np.full(len(xyz), minimum_scale)
    )
    spread = np.sqrt(np.maximum(eigenvalues[:, :2], 1e-12))
    uncertainty = np.asarray(position_sigma, dtype=np.float64)[:, None]
    scales = np.maximum(
        0.50 * spread,
        np.maximum(0.45 * nearest[:, None], 0.75 * uncertainty),
    )
    scales = np.clip(scales, minimum_scale, maximum_scale).astype(
        np.float32
    )
    normals = frames[:, 2].astype(np.float32)
    return scales, quaternion, normals


def _chart_filepaths(camera_path: Path) -> list[Path]:
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    filepaths = payload.get("filepaths")
    if not isinstance(filepaths, list):
        raise RuntimeError(f"Chart cameras lack filepaths: {camera_path}")
    return [Path(value) for value in filepaths]


def _sample_chart_seeds(
    chart_path: Path,
    chart_cameras: Path,
    *,
    dataset: Path,
    tree_mask_pickle: Path,
    rigid_reference: np.ndarray,
    per_chart: int,
    maximum_total: int,
    consensus_radius: float,
    seed: int,
) -> dict[str, np.ndarray]:
    lookup = CambridgeMaskLookup(
        dataset, tree_mask_pickle, mask_indices=[0, 1, 2, 3]
    )
    files = _chart_filepaths(chart_cameras)
    rng = np.random.default_rng(seed)
    selected_xyz = []
    selected_rgb = []
    selected_sigma = []
    selected_confidence = []
    with np.load(chart_path, allow_pickle=False) as charts:
        points = charts["pts"]
        confidence = charts["confs"]
        active = charts.get(
            "quality_selection_active",
            np.ones(points.shape[0], dtype=bool),
        ).astype(bool)
        active &= charts.get(
            "alignment_gate_valid",
            np.ones(points.shape[0], dtype=bool),
        ).astype(bool)
        reference_mask = charts.get(
            "alignment_reference_mask",
            np.ones(points.shape[:3], dtype=bool),
        ).astype(bool)
        if len(files) != len(points):
            raise RuntimeError("Chart camera/image count mismatch")
        reference_tree = cKDTree(rigid_reference) if len(rigid_reference) else None
        for chart_index in np.nonzero(active)[0]:
            pointmap = points[chart_index]
            conf = confidence[chart_index]
            height, width = conf.shape
            image_name = files[chart_index].name
            key = lookup.source_name_for(image_name)
            channels = lookup.masks[key]
            resized = []
            for value in channels[:4]:
                value = torch.nn.functional.interpolate(
                    value.detach().to(dtype=torch.float32)[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0].numpy() > 0.5
                resized.append(value)
            rigid = resized[0] & resized[1] & resized[2] & resized[3]
            valid = (
                rigid
                & reference_mask[chart_index]
                & np.isfinite(pointmap).all(axis=-1)
                & (np.linalg.norm(pointmap, axis=-1) > 1e-5)
                & np.isfinite(conf)
                & (conf > 0)
            )
            flat = np.flatnonzero(valid)
            if not len(flat):
                continue
            limit = min(int(per_chart), len(flat))
            # Keep the strongest half and a deterministic random half so
            # facade coverage is not reduced to only high-texture corners.
            strong_count = max(1, limit // 2)
            strongest = flat[
                np.argpartition(conf.reshape(-1)[flat], -strong_count)[
                    -strong_count:
                ]
            ]
            remainder = np.setdiff1d(flat, strongest, assume_unique=False)
            random_count = limit - len(strongest)
            sampled = (
                rng.choice(remainder, random_count, replace=False)
                if random_count and len(remainder) >= random_count
                else remainder[:random_count]
            )
            chosen = np.concatenate([strongest, sampled])
            xyz = pointmap.reshape(-1, 3)[chosen].astype(np.float32)
            conf_value = conf.reshape(-1)[chosen].astype(np.float32)
            if reference_tree is not None:
                cross_distance = reference_tree.query(xyz, k=1)[0]
                high = conf_value >= np.quantile(conf_value, 0.80)
                supported = (
                    (cross_distance <= float(consensus_radius))
                    | (
                        high
                        & (cross_distance <= 2.0 * float(consensus_radius))
                    )
                )
                xyz = xyz[supported]
                conf_value = conf_value[supported]
                chosen = chosen[supported]
            if not len(xyz):
                continue
            image_path = files[chart_index]
            if not image_path.is_file():
                image_path = dataset / "images" / image_name
            image = Image.open(image_path).convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            color = np.asarray(image, dtype=np.float32).reshape(-1, 3)[
                chosen
            ] / 255.0
            # Confidence is used only as an uncertainty proxy and remains
            # source-labelled; it is not converted into a universal truth.
            normalized = conf_value / max(
                float(np.median(conf_value)), 1e-6
            )
            sigma = np.clip(0.025 / np.sqrt(normalized), 0.006, 0.08)
            selected_xyz.append(xyz)
            selected_rgb.append(color.astype(np.float32))
            selected_sigma.append(sigma.astype(np.float32))
            selected_confidence.append(normalized.astype(np.float32))
    if not selected_xyz:
        return {
            "xyz": np.empty((0, 3), dtype=np.float32),
            "rgb": np.empty((0, 3), dtype=np.float32),
            "sigma": np.empty(0, dtype=np.float32),
            "confidence": np.empty(0, dtype=np.float32),
        }
    xyz = np.concatenate(selected_xyz)
    rgb = np.concatenate(selected_rgb)
    sigma = np.concatenate(selected_sigma)
    confidence = np.concatenate(selected_confidence)
    if len(xyz) > int(maximum_total):
        order = np.argsort(confidence)[-int(maximum_total) :]
        xyz, rgb, sigma, confidence = (
            xyz[order],
            rgb[order],
            sigma[order],
            confidence[order],
        )
    return {
        "xyz": xyz,
        "rgb": rgb,
        "sigma": sigma,
        "confidence": confidence,
    }


def build_surface_seed(
    evidence_store: Path,
    output: Path,
    *,
    minimum_rigid_probability: float = 0.70,
    maximum_canopy_probability: float = 0.15,
    consensus_radius: float = 0.10,
    dedup_radius: float = 0.012,
    chart_seeds_per_view: int = 2000,
    maximum_chart_seeds: int = 80_000,
    seed: int = 991,
) -> dict[str, Any]:
    store = load_evidence_store(evidence_store)
    colmap = _load_tracks(artifact_path(store, "colmap_tracks"))
    colmap_keep = _selection_mask(
        colmap,
        minimum_rigid_probability=minimum_rigid_probability,
        maximum_canopy_probability=maximum_canopy_probability,
        minimum_observations=2,
        maximum_reprojection_error=2.5,
    )
    xyz_parts = [colmap["xyz"][colmap_keep]]
    rgb_parts = [colmap["rgb"][colmap_keep].astype(np.float32) / 255.0]
    sigma_parts = [
        np.sqrt(
            colmap["position_covariance_diag"][colmap_keep].mean(axis=1)
        )
    ]
    confidence_parts = [
        colmap["role_probabilities"][colmap_keep, ROLE_RIGID]
    ]
    track_parts = [colmap["track_id"][colmap_keep]]
    source_parts = [
        np.full(int(colmap_keep.sum()), SOURCE_COLMAP, dtype=np.int8)
    ]
    source_counts = {"colmap_rigid": int(colmap_keep.sum())}

    mast3r_path = artifact_path(
        store, "mast3r_tracks", required=False
    )
    if mast3r_path is not None:
        mast3r = _load_tracks(mast3r_path)
        mast3r_keep = _selection_mask(
            mast3r,
            minimum_rigid_probability=max(
                minimum_rigid_probability, 0.75
            ),
            maximum_canopy_probability=maximum_canopy_probability,
            # Posed-MASt3R's sparse export stores one source camera per
            # point.  Cross-view validity is supplied by the independent
            # COLMAP-neighbour consensus below; requiring two rows here would
            # silently discard the whole MASt3R source.
            minimum_observations=1,
            maximum_reprojection_error=2.5,
        )
        candidate = mast3r["xyz"][mast3r_keep]
        reference = xyz_parts[0]
        if len(candidate) and len(reference):
            distance = cKDTree(reference).query(candidate, k=1)[0]
            candidate_support = distance <= min(
                float(consensus_radius), 0.05
            )
        else:
            candidate_support = np.ones(len(candidate), dtype=bool)
        mast3r_indices = np.flatnonzero(mast3r_keep)[candidate_support]
        novel = _deduplicate(
            mast3r["xyz"][mast3r_indices],
            np.concatenate(xyz_parts),
            radius=dedup_radius,
        )
        mast3r_indices = mast3r_indices[novel]
        xyz_parts.append(mast3r["xyz"][mast3r_indices])
        rgb_parts.append(
            mast3r["rgb"][mast3r_indices].astype(np.float32) / 255.0
        )
        sigma_parts.append(
            np.sqrt(
                mast3r["position_covariance_diag"][
                    mast3r_indices
                ].mean(axis=1)
            )
        )
        confidence_parts.append(
            mast3r["role_probabilities"][
                mast3r_indices, ROLE_RIGID
            ]
        )
        track_parts.append(mast3r["track_id"][mast3r_indices])
        source_parts.append(
            np.full(
                len(mast3r_indices), SOURCE_MAST3R, dtype=np.int8
            )
        )
        source_counts["mast3r_rigid"] = int(len(mast3r_indices))

    chart_path = artifact_path(
        store, "chart_geometry", required=False
    )
    chart_cameras = artifact_path(
        store, "chart_cameras", required=False
    )
    if chart_path is not None and chart_cameras is not None:
        chart = _sample_chart_seeds(
            chart_path,
            chart_cameras,
            dataset=Path(store["dataset"]),
            tree_mask_pickle=Path(
                json.loads(
                    Path(store["semantic_contract"]).read_text(
                        encoding="utf-8"
                    )
                )["tree_mask_pickle"]
            ),
            rigid_reference=np.concatenate(xyz_parts),
            per_chart=chart_seeds_per_view,
            maximum_total=maximum_chart_seeds,
            consensus_radius=consensus_radius,
            seed=seed,
        )
        novel = _deduplicate(
            chart["xyz"],
            np.concatenate(xyz_parts),
            radius=dedup_radius,
        )
        for key in chart:
            chart[key] = chart[key][novel]
        xyz_parts.append(chart["xyz"])
        rgb_parts.append(chart["rgb"])
        sigma_parts.append(chart["sigma"])
        confidence_parts.append(
            np.clip(chart["confidence"] / 2.0, 0.05, 1.0)
        )
        track_parts.append(
            np.full(len(chart["xyz"]), -1, dtype=np.int64)
        )
        source_parts.append(
            np.full(len(chart["xyz"]), SOURCE_CHART, dtype=np.int8)
        )
        source_counts["chart_rigid_residual"] = int(len(chart["xyz"]))

    xyz = np.concatenate(xyz_parts).astype(np.float32)
    rgb = np.concatenate(rgb_parts).astype(np.float32)
    sigma = np.concatenate(sigma_parts).astype(np.float32)
    confidence = np.concatenate(confidence_parts).astype(np.float32)
    track_id = np.concatenate(track_parts).astype(np.int64)
    source_type = np.concatenate(source_parts).astype(np.int8)
    scales, quaternions, normals = _surface_frames(xyz, sigma)
    initial_opacity = np.choose(
        source_type,
        np.asarray([0.08, 0.035, 0.015], dtype=np.float32),
    ).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        version=np.asarray(INITIALIZATION_VERSION),
        xyz=xyz,
        rgb=rgb,
        scales=scales,
        quaternions=quaternions,
        normals=normals,
        initial_opacity=initial_opacity,
        geometry_confidence=confidence,
        position_sigma=sigma,
        source_type=source_type,
        track_id=track_id,
    )
    summary = {
        "version": INITIALIZATION_VERSION,
        "evidence_hash": store["evidence_hash"],
        "surface_count": len(xyz),
        "source_counts": source_counts,
        "canopy_surface_seed_count": 0,
        "historical_trained_ply_used": False,
        "all_real_rgb_initialization_used": False,
        "scale_median": np.median(scales, axis=0).tolist(),
        "position_sigma_median": float(np.median(sigma)),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def sparse_rigid_depth_maps(
    views: list[dict],
    rigid_xyz: np.ndarray,
    *,
    resolution_scale: float = 0.35,
    dilation_pixels: int = 5,
) -> dict[int, np.ndarray]:
    """Conservative sparse rigid z-buffer used only as an occlusion prior."""
    output = {}
    xyz = np.asarray(rigid_xyz, dtype=np.float64)
    for view in views:
        width = max(32, int(round(view["width"] * resolution_scale)))
        height = max(32, int(round(view["height"] * resolution_scale)))
        camera_xyz = (
            xyz @ np.asarray(view["rotation"]).T
            + np.asarray(view["translation"])[None]
        )
        depth = camera_xyz[:, 2]
        u = (
            camera_xyz[:, 0]
            / np.maximum(depth, 1e-8)
            * view["fx"]
            + view["cx"]
        )
        v = (
            camera_xyz[:, 1]
            / np.maximum(depth, 1e-8)
            * view["fy"]
            + view["cy"]
        )
        columns = np.floor(u / view["width"] * width).astype(np.int64)
        rows = np.floor(v / view["height"] * height).astype(np.int64)
        valid = (
            (depth > 0.05)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        buffer = np.full(height * width, np.inf, dtype=np.float32)
        linear = rows[valid] * width + columns[valid]
        np.minimum.at(buffer, linear, depth[valid].astype(np.float32))
        buffer = buffer.reshape(height, width)
        if dilation_pixels > 1:
            buffer = minimum_filter(
                buffer,
                size=int(dilation_pixels),
                mode="constant",
                cval=np.inf,
            )
        output[int(view["image_id"])] = buffer
    return output


def _merge_foliage(
    hull: dict[str, np.ndarray],
    tracks: list[dict],
    track_instances: np.ndarray,
    *,
    skeleton_linearity: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    xyz = np.stack([point["xyz"] for point in tracks]).astype(np.float32)
    rgb = (
        np.stack([point["rgb"] for point in tracks]).astype(np.float32)
        / 255.0
    )
    scales, quaternions, linearity = _track_frames(
        xyz, 0.008, 0.035, 0.10
    )
    track_count = len(tracks)
    capacity = hull["support_camera_ids"].shape[1]
    support_camera_ids = np.full(
        (track_count, capacity), -1, dtype=np.int32
    )
    for index, point in enumerate(tracks):
        values = np.unique(point["tree_image_ids"])[:capacity]
        support_camera_ids[index, : len(values)] = values
    error = np.asarray(
        [point["error"] for point in tracks], dtype=np.float32
    )
    covariance_scale = np.maximum(0.008, 0.012 + 0.01 * error)
    position_covariance = np.eye(3, dtype=np.float32)[None] * (
        covariance_scale[:, None, None] ** 2
    )
    layer_role = np.where(
        linearity >= float(skeleton_linearity), 1, 0
    ).astype(np.int8)
    values = {
        "centers": xyz,
        "colors": rgb,
        "scales": scales,
        "opacities": np.full(
            (track_count, 1), 0.025, dtype=np.float32
        ),
        "quaternions": quaternions,
        "primitive_role": np.ones(track_count, dtype=np.int8),
        "layer_role": layer_role,
        "track_id": np.asarray(
            [point["id"] for point in tracks], dtype=np.int64
        ),
        "tree_instance_id": track_instances.astype(np.int32),
        "initialization_source": np.ones(
            track_count, dtype=np.int8
        ),
        "occupancy_probability": np.asarray(
            [point["tree_fraction"] for point in tracks],
            dtype=np.float32,
        ),
        "position_covariance": position_covariance,
        "reprojection_error": error,
        "track_linearity": linearity,
        "support_camera_ids": support_camera_ids,
        "support_view_count": np.asarray(
            [len(point["tree_image_ids"]) for point in tracks],
            dtype=np.int16,
        ),
        "support_sequence_count": np.asarray(
            [point["tree_sequence_count"] for point in tracks],
            dtype=np.int16,
        ),
    }
    merged = {}
    for key, track_value in values.items():
        if key not in hull:
            raise RuntimeError(f"Canopy hull lacks required field {key}")
        merged[key] = np.concatenate(
            [hull[key], track_value], axis=0
        )
    for key, value in hull.items():
        if (
            key in merged
            or not isinstance(value, np.ndarray)
            or len(value) != len(hull["centers"])
        ):
            continue
        fill = np.zeros(
            (track_count,) + value.shape[1:], dtype=value.dtype
        )
        merged[key] = np.concatenate([value, fill], axis=0)
    return merged, {
        "visual_hull": int(len(hull["centers"])),
        "tree_tracks": track_count,
        "static_skeleton": int((layer_role == 1).sum()),
        "canonical_track_anchors": int((layer_role == 0).sum()),
        "tree_instances": int(track_instances.max()) + 1,
    }


def build_foliage_seed(
    evidence_store: Path,
    surface_seed: Path,
    output: Path,
    *,
    voxel_size: float = 0.12,
    maximum_voxels: int = 400_000,
    selected_view_count: int = 64,
    minimum_track_observations: int = 3,
    minimum_track_sequences: int = 2,
    maximum_reprojection_error: float = 2.0,
    skeleton_linearity: float = 1.8,
    seed: int = 73,
) -> dict[str, Any]:
    store = load_evidence_store(evidence_store)
    dataset = Path(store["dataset"])
    semantic = json.loads(
        Path(store["semantic_contract"]).read_text(encoding="utf-8")
    )
    tree_mask = Path(semantic["tree_mask_pickle"])
    with np.load(surface_seed, allow_pickle=False) as surface:
        rigid_xyz = surface["xyz"].astype(np.float32)
    sparse = dataset / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points = read_points3d_binary_with_tracks(sparse / "points3D.bin")
    masks = CambridgeMaskLookup(
        dataset, tree_mask, mask_indices=[0, 1, 2, 3]
    )
    tracks = semantic_tree_tracks(
        points,
        images,
        cameras,
        masks,
        minimum_tree_observations=minimum_track_observations,
        minimum_sequences=minimum_track_sequences,
        maximum_reprojection_error=maximum_reprojection_error,
    )
    if not tracks:
        raise RuntimeError("No multi-view semantic tree tracks survived")
    track_xyz = np.stack([point["xyz"] for point in tracks])
    instances = cluster_tree_instances(track_xyz)
    view_records = [
        _camera_record(image, cameras[image["camera_id"]], masks)
        for image in images.values()
    ]
    selected = greedy_diverse_views(
        view_records,
        limit=selected_view_count,
        minimum_center_distance=0.75,
    )
    rigid_depth = sparse_rigid_depth_maps(selected, rigid_xyz)
    hull = build_instance_aware_canopy_volume(
        tracks,
        images,
        cameras,
        masks,
        rigid_depth_maps=rigid_depth,
        selected_views=selected,
        voxel_size=voxel_size,
        maximum_voxels=maximum_voxels,
        seed=seed,
    )
    selected_views = hull.pop("selected_views")
    candidate_count = int(hull.pop("candidate_count"))
    rigid_depth_view_count = int(hull.pop("rigid_depth_view_count"))
    tree_instance_count = int(hull.pop("tree_instance_count"))
    merged, counts = _merge_foliage(
        hull,
        tracks,
        instances,
        skeleton_linearity=skeleton_linearity,
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "geometry_version": (
            "unified_evidence_instance_occlusion_ray_depth_v1"
        ),
        "audit": {
            "protocol": INITIALIZATION_VERSION,
            "evidence_hash": store["evidence_hash"],
            "candidate_voxels": candidate_count,
            "rigid_depth_view_count": rigid_depth_view_count,
            "tree_instance_count": tree_instance_count,
            "positive_negative_unknown_evidence": True,
            "real_ray_depth_posterior": True,
            "sparse_rigid_occlusion_zbuffer": True,
            "historical_trained_ply_used": False,
            **counts,
            "selected_views": [
                {
                    "image_id": int(view["image_id"]),
                    "image_name": str(view["image_name"]),
                    "sequence_id": str(view["sequence_id"]),
                }
                for view in selected_views
            ],
        },
        **{
            key: torch.from_numpy(value)
            for key, value in merged.items()
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = {
        **payload["audit"],
        "total_count": int(len(merged["centers"])),
        "support_views_median": float(
            np.median(merged["support_view_count"])
        ),
        "support_sequences_median": float(
            np.median(merged["support_sequence_count"])
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
