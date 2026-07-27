"""Role-aware initialization from a unified outdoor evidence store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from matcha.cambridge_masks import CambridgeMaskLookup
from matcha.see3d_geometry import robust_align_inverse_depth
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
    quaternion_to_rotation,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary_with_tracks,
    semantic_tree_tracks,
)
from outdoor.foliage_view_graph import (
    greedy_diverse_views,
    sequence_balanced_diverse_views,
)
from scripts.augment_foliage_seed_with_sfm_tracks import _track_frames


INITIALIZATION_VERSION = (
    "outdoor-role-aware-initialization-v8-exact-observation-rays"
)

SOURCE_COLMAP = 0
SOURCE_MAST3R = 1
SOURCE_CHART = 2
SOURCE_DAV2 = 3


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
    selected_chart_id = []
    selected_uv = []
    selected_scales = []
    selected_quaternions = []
    selected_normals = []
    with np.load(chart_path, allow_pickle=False) as charts:
        scale_factor = float(charts["scale_factor"])
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise RuntimeError(
                f"Invalid MAtCha chart scale_factor {scale_factor}"
            )
        # MAtCha optimizes its atlas in a normalized scene.  Cambridge
        # cameras, MASt3R tracks, and the final Teacher stay in the original
        # fixed-camera world, so chart points must cross this boundary
        # explicitly.  Omitting this conversion made valid chart evidence
        # look like near-origin floaters.
        points = charts["pts"] / scale_factor
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
            sigma = np.clip(
                (0.025 / scale_factor) / np.sqrt(normalized),
                0.012,
                0.35,
            )
            selected_xyz.append(xyz)
            selected_rgb.append(color.astype(np.float32))
            selected_sigma.append(sigma.astype(np.float32))
            selected_confidence.append(normalized.astype(np.float32))
            selected_chart_id.append(
                np.full(len(xyz), int(chart_index), dtype=np.int32)
            )
            rows, columns = np.unravel_index(chosen, (height, width))
            left = pointmap[rows, np.maximum(columns - 1, 0)]
            right = pointmap[rows, np.minimum(columns + 1, width - 1)]
            up = pointmap[np.maximum(rows - 1, 0), columns]
            down = pointmap[np.minimum(rows + 1, height - 1), columns]
            tangent_u = right - left
            tangent_v = down - up
            axis_u = tangent_u / np.maximum(
                np.linalg.norm(tangent_u, axis=1, keepdims=True), 1e-8
            )
            tangent_v = tangent_v - (
                tangent_v * axis_u
            ).sum(axis=1, keepdims=True) * axis_u
            axis_v = tangent_v / np.maximum(
                np.linalg.norm(tangent_v, axis=1, keepdims=True), 1e-8
            )
            normal = np.cross(axis_u, axis_v)
            normal /= np.maximum(
                np.linalg.norm(normal, axis=1, keepdims=True), 1e-8
            )
            frames = np.stack([axis_u, axis_v, normal], axis=2)
            finite_frame = np.isfinite(frames).all(axis=(1, 2))
            frames[~finite_frame] = np.eye(3, dtype=np.float32)
            xyzw = Rotation.from_matrix(frames).as_quat().astype(np.float32)
            quaternion = np.column_stack(
                [xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]]
            )
            footprint = np.column_stack(
                [
                    0.55 * np.linalg.norm(tangent_u, axis=1),
                    0.55 * np.linalg.norm(tangent_v, axis=1),
                ]
            )
            footprint = np.clip(footprint, 0.003, 0.06).astype(np.float32)
            selected_uv.append(
                np.column_stack(
                    [
                        (columns + 0.5) / float(width),
                        (rows + 0.5) / float(height),
                    ]
                ).astype(np.float32)
            )
            selected_scales.append(footprint)
            selected_quaternions.append(quaternion)
            selected_normals.append(normal.astype(np.float32))
    if not selected_xyz:
        return {
            "xyz": np.empty((0, 3), dtype=np.float32),
            "rgb": np.empty((0, 3), dtype=np.float32),
            "sigma": np.empty(0, dtype=np.float32),
            "confidence": np.empty(0, dtype=np.float32),
            "chart_id": np.empty(0, dtype=np.int32),
            "uv": np.empty((0, 2), dtype=np.float32),
            "scales": np.empty((0, 2), dtype=np.float32),
            "quaternions": np.empty((0, 4), dtype=np.float32),
            "normals": np.empty((0, 3), dtype=np.float32),
        }
    xyz = np.concatenate(selected_xyz)
    rgb = np.concatenate(selected_rgb)
    sigma = np.concatenate(selected_sigma)
    confidence = np.concatenate(selected_confidence)
    chart_id = np.concatenate(selected_chart_id)
    uv = np.concatenate(selected_uv)
    scales = np.concatenate(selected_scales)
    quaternions = np.concatenate(selected_quaternions)
    normals = np.concatenate(selected_normals)
    if len(xyz) > int(maximum_total):
        order = np.argsort(confidence)[-int(maximum_total) :]
        xyz, rgb, sigma, confidence, chart_id, uv, scales, quaternions, normals = (
            xyz[order],
            rgb[order],
            sigma[order],
            confidence[order],
            chart_id[order],
            uv[order],
            scales[order],
            quaternions[order],
            normals[order],
        )
    return {
        "xyz": xyz,
        "rgb": rgb,
        "sigma": sigma,
        "confidence": confidence,
        "chart_id": chart_id,
        "uv": uv,
        "scales": scales,
        "quaternions": quaternions,
        "normals": normals,
    }


def build_surface_seed(
    evidence_store: Path,
    output: Path,
    *,
    minimum_rigid_probability: float = 0.70,
    maximum_canopy_probability: float = 0.15,
    consensus_radius: float = 0.10,
    dedup_radius: float = 0.012,
    chart_seeds_per_view: int = 3000,
    maximum_chart_seeds: int = 120_000,
    seed: int = 991,
) -> dict[str, Any]:
    store = load_evidence_store(evidence_store)
    mast3r_only = store.get("geometry_source") == "mast3r_only"
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    track_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    chart_id_parts: list[np.ndarray] = []
    chart_uv_parts: list[np.ndarray] = []
    source_counts: dict[str, int] = {}

    if not mast3r_only:
        colmap = _load_tracks(artifact_path(store, "colmap_tracks"))
        colmap_keep = _selection_mask(
            colmap,
            minimum_rigid_probability=minimum_rigid_probability,
            maximum_canopy_probability=maximum_canopy_probability,
            minimum_observations=2,
            maximum_reprojection_error=2.5,
        )
        xyz_parts.append(colmap["xyz"][colmap_keep])
        rgb_parts.append(
            colmap["rgb"][colmap_keep].astype(np.float32) / 255.0
        )
        sigma_parts.append(
            np.sqrt(
                colmap["position_covariance_diag"][colmap_keep].mean(axis=1)
            )
        )
        confidence_parts.append(
            colmap["role_probabilities"][colmap_keep, ROLE_RIGID]
        )
        track_parts.append(colmap["track_id"][colmap_keep])
        source_parts.append(
            np.full(int(colmap_keep.sum()), SOURCE_COLMAP, dtype=np.int8)
        )
        chart_id_parts.append(
            np.full(int(colmap_keep.sum()), -1, dtype=np.int32)
        )
        chart_uv_parts.append(
            np.full((int(colmap_keep.sum()), 2), np.nan, dtype=np.float32)
        )
        source_counts["colmap_rigid"] = int(colmap_keep.sum())

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
            # The new mainline requires genuine ragged multi-view tracks.
            # A one-camera MASt3R sparse export cannot pass this selection.
            minimum_observations=2 if mast3r_only else 1,
            maximum_reprojection_error=2.0 if mast3r_only else 2.5,
        )
        candidate = mast3r["xyz"][mast3r_keep]
        reference = (
            np.concatenate(xyz_parts)
            if xyz_parts
            else np.empty((0, 3), dtype=np.float32)
        )
        if not mast3r_only and len(candidate) and len(reference):
            distance = cKDTree(reference).query(candidate, k=1)[0]
            candidate_support = distance <= min(
                float(consensus_radius), 0.05
            )
        else:
            candidate_support = np.ones(len(candidate), dtype=bool)
        mast3r_indices = np.flatnonzero(mast3r_keep)[candidate_support]
        novel = _deduplicate(
            mast3r["xyz"][mast3r_indices],
            reference,
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
        chart_id_parts.append(
            np.full(len(mast3r_indices), -1, dtype=np.int32)
        )
        chart_uv_parts.append(
            np.full((len(mast3r_indices), 2), np.nan, dtype=np.float32)
        )
        source_counts["mast3r_rigid"] = int(len(mast3r_indices))
    if mast3r_only and not xyz_parts:
        raise RuntimeError(
            "MASt3R-only initialization has no valid multi-view rigid tracks"
        )

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
        # Negative ids below -1 are permanent Chart evidence identities.
        # They are deliberately outside the SfM track namespace and survive
        # render-primitive split/retire through metadata inheritance.
        track_parts.append(
            -(np.arange(len(chart["xyz"]), dtype=np.int64) + 2)
        )
        source_parts.append(
            np.full(len(chart["xyz"]), SOURCE_CHART, dtype=np.int8)
        )
        chart_id_parts.append(chart["chart_id"].astype(np.int32))
        chart_uv_parts.append(chart["uv"].astype(np.float32))
        source_counts["chart_rigid_residual"] = int(len(chart["xyz"]))

    xyz = np.concatenate(xyz_parts).astype(np.float32)
    rgb = np.concatenate(rgb_parts).astype(np.float32)
    sigma = np.concatenate(sigma_parts).astype(np.float32)
    confidence = np.concatenate(confidence_parts).astype(np.float32)
    track_id = np.concatenate(track_parts).astype(np.int64)
    source_type = np.concatenate(source_parts).astype(np.int8)
    chart_id = np.concatenate(chart_id_parts).astype(np.int32)
    chart_uv = np.concatenate(chart_uv_parts).astype(np.float32)
    scales = np.empty((len(xyz), 2), dtype=np.float32)
    quaternions = np.empty((len(xyz), 4), dtype=np.float32)
    normals = np.empty((len(xyz), 3), dtype=np.float32)
    free = source_type != SOURCE_CHART
    if bool(free.any()):
        scales[free], quaternions[free], normals[free] = _surface_frames(
            xyz[free], sigma[free], maximum_scale=0.08
        )
    chart_rows = source_type == SOURCE_CHART
    if bool(chart_rows.any()):
        scales[chart_rows] = chart["scales"]
        quaternions[chart_rows] = chart["quaternions"]
        normals[chart_rows] = chart["normals"]
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
        chart_id=chart_id,
        chart_uv=chart_uv,
    )
    summary = {
        "version": INITIALIZATION_VERSION,
        "evidence_hash": store["evidence_hash"],
        "surface_count": len(xyz),
        "source_counts": source_counts,
        "canopy_surface_seed_count": 0,
        "historical_trained_ply_used": False,
        "all_real_rgb_initialization_used": False,
        "geometry_source": store.get("geometry_source", "legacy_mixed"),
        "colmap_points_or_tracks_used": not mast3r_only,
        "scale_median": np.median(scales, axis=0).tolist(),
        "position_sigma_median": float(np.median(sigma)),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _mast3r_tree_tracks(
    archive_path: Path,
    images: dict[int, dict],
    *,
    minimum_track_observations: int,
    maximum_reprojection_error: float,
) -> list[dict[str, Any]]:
    """Adapt native MASt3R track observations to the canopy-volume contract."""
    by_stem = {
        Path(str(image["name"])).stem: int(image_id)
        for image_id, image in images.items()
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["camera_names"]]
        mapped_ids = np.asarray(
            [by_stem.get(Path(name).stem, -1) for name in names],
            dtype=np.int32,
        )
        offsets = archive["observation_offsets"]
        observation_cameras = archive["observation_camera_indices"]
        observation_pixels = archive["observation_pixels"]
        observation_depth = archive["observation_camera_depth"]
        camera_image_sizes = archive["camera_image_sizes"]
        keep = (
            (
                archive["role_probabilities"][:, ROLE_CANOPY]
                >= 0.55
            )
            & (
                archive["valid_observation_count"]
                >= int(minimum_track_observations)
            )
            & (
                archive["reprojection_error"]
                <= float(maximum_reprojection_error)
            )
        )
        result: list[dict[str, Any]] = []
        for index in np.flatnonzero(keep):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            camera_rows = observation_cameras[begin:end]
            mapped_observations = mapped_ids[camera_rows]
            valid_observation = mapped_observations >= 0
            image_ids = np.unique(
                mapped_observations[valid_observation]
            )
            if len(image_ids) < int(minimum_track_observations):
                continue
            pixels = observation_pixels[begin:end][valid_observation]
            sizes = camera_image_sizes[camera_rows][valid_observation]
            result.append(
                {
                    "id": int(archive["track_id"][index]),
                    "xyz": archive["xyz"][index].astype(np.float64),
                    "rgb": archive["rgb"][index].astype(np.uint8),
                    "error": float(
                        archive["reprojection_error"][index]
                    ),
                    "image_ids": image_ids,
                    "point2d_indices": np.full(
                        len(image_ids), -1, dtype=np.int64
                    ),
                    "tree_image_ids": image_ids,
                    "observation_camera_ids": mapped_observations[
                        valid_observation
                    ].astype(np.int32),
                    "observation_uv": (
                        pixels.astype(np.float32)
                        / np.maximum(sizes.astype(np.float32), 1.0)
                    ).astype(np.float32),
                    "observation_depth": observation_depth[
                        begin:end
                    ][valid_observation].astype(np.float32),
                    "tree_sequence_count": int(
                        archive["sequence_count"][index]
                    ),
                    "tree_fraction": float(
                        archive["role_probabilities"][
                            index, ROLE_CANOPY
                        ]
                    ),
                }
            )
    return result


def _cross_view_tree_hull_anchors(
    local_tracks: list[dict[str, Any]],
    images: dict[int, dict],
    cameras: dict[int, dict],
    masks: CambridgeMaskLookup,
    *,
    maximum_reprojection_error: float,
) -> list[dict[str, Any]]:
    """Validate local foliage anchors against all fixed-camera tree masks.

    This is a visual-hull support operation, not an invented static feature
    correspondence.  The returned cross-view support may seed canonical crown
    occupancy, but the original sequence-local track remains the only dynamic
    correspondence and neither representation is exported as a localization
    landmark.
    """
    all_image_ids = np.asarray(sorted(images), dtype=np.int64)
    candidates = [
        {
            **track,
            "image_ids": all_image_ids,
            "point2d_indices": np.full(
                len(all_image_ids), -1, dtype=np.int64
            ),
        }
        for track in local_tracks
    ]
    return semantic_tree_tracks(
        candidates,
        images,
        cameras,
        masks,
        minimum_tree_observations=3,
        minimum_sequences=2,
        minimum_tree_fraction=0.60,
        maximum_reprojection_error=maximum_reprojection_error,
    )


def _chart_foliage_samples(
    store: dict,
    images: dict[int, dict],
    masks: CambridgeMaskLookup,
    *,
    samples_per_view: int = 1_600,
    maximum_samples: int = 24_000,
    voxel_size: float = 0.04,
) -> list[dict[str, Any]]:
    """Fill sequence-local leaf coverage where no pairwise track can exist.

    The aligned chart archive deliberately zeros canopy pixels because those
    pixels are invalid for rigid chart alignment.  The raw MASt3R pointmaps,
    however, retain their world points and confidence there.  We therefore
    use the chart only to define an active, fixed-camera-aligned view and a
    conservative nearby correction field; canopy geometry comes from the raw
    pointmap and is always assigned to the dynamic leaf branch.
    """
    chart_path = artifact_path(store, "chart_geometry", required=False)
    camera_path = artifact_path(store, "chart_cameras", required=False)
    if chart_path is None or camera_path is None:
        return []
    camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
    filepaths = [Path(value) for value in camera_payload["filepaths"]]
    pointmap_index_path = artifact_path(
        store, "mast3r_pointmap_index", required=False
    )
    if pointmap_index_path is not None:
        pointmap_records = json.loads(
            pointmap_index_path.read_text(encoding="utf-8")
        )["records"]
    else:
        # Legacy evidence stores are accepted for regression tests only.
        pointmap_root = camera_path.parent / "pointmaps"
        if not pointmap_root.is_dir():
            return []
        pointmap_records = {
            path.stem: {"path": str(path)}
            for path in pointmap_root.glob("*.json")
        }
    by_stem = {
        Path(str(image["name"])).stem: int(image_id)
        for image_id, image in images.items()
    }
    result: list[dict[str, Any]] = []
    # Deduplicate inside a camera only. Global voxel deduplication erased the
    # later traversal's independent depth posterior and made cross-sequence
    # support impossible by construction.
    occupied: set[tuple[int, int, int, int]] = set()

    def load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Decode point/conf arrays without materializing the huge RGB list."""
        raw = path.read_bytes()
        point_marker = b'"points":'
        confidence_marker = b'"confs":'
        point_start = raw.find(point_marker)
        confidence_start = raw.find(confidence_marker)
        if point_start < 0 or confidence_start < 0:
            raise RuntimeError(f"Malformed MASt3R pointmap: {path}")
        point_start += len(point_marker)
        confidence_value_start = confidence_start + len(
            confidence_marker
        )
        payload_end = raw.rfind(b"}")
        point_payload = raw[point_start:confidence_start].rstrip()
        if point_payload.endswith(b","):
            point_payload = point_payload[:-1].rstrip()
        # ``json.loads`` creates hundreds of thousands of boxed Python floats
        # per view.  Parsing the numeric payload directly keeps peak memory
        # bounded across all active charts.
        bracket_table = bytes.maketrans(b"[]", b"  ")
        point_values = np.fromstring(
            point_payload.translate(bracket_table),
            dtype=np.float32,
            sep=",",
        )
        confidence_payload = raw[
            confidence_value_start:payload_end
        ].strip()
        confidence_values = np.fromstring(
            confidence_payload.translate(bracket_table),
            dtype=np.float32,
            sep=",",
        )
        pixel_count = int(confidence_values.size)
        if point_values.size != 3 * pixel_count:
            raise RuntimeError(
                f"Point/conf size mismatch in MASt3R pointmap {path}"
            )
        # Current MASt3R pointmaps are written with their native 16:9 shape.
        # Resolve the exact height through the chart tensor below; retain a
        # flat representation here to avoid guessing from JSON formatting.
        confidence_array = confidence_values
        point_array = point_values.reshape(pixel_count, 3)
        return point_array, confidence_array

    with np.load(chart_path, allow_pickle=False) as archive:
        chart_points = archive["pts"]
        chart_confidence = archive["confs"]
        scale_factor = float(archive["scale_factor"])
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise RuntimeError(
                f"Invalid MAtCha chart scale_factor {scale_factor}"
            )
        active = archive.get(
            "quality_selection_active",
            np.ones(len(chart_points), dtype=bool),
        ).astype(bool)
        active &= archive.get(
            "alignment_gate_valid",
            np.ones(len(chart_points), dtype=bool),
        ).astype(bool)
        # Rigid Chart alignment status must not erase a real sequence-local
        # canopy observation. Inactive Charts are excluded from the building
        # surface branch, but their raw MASt3R pointmaps remain useful for
        # dynamic leaves with lower confidence and sequence-local visibility.
        for chart_index in range(len(chart_points)):
            image_path = filepaths[int(chart_index)]
            image_id = by_stem.get(image_path.stem)
            if image_id is None:
                continue
            pointmap_record = pointmap_records.get(image_path.stem)
            if pointmap_record is None:
                continue
            pointmap_path = Path(pointmap_record["path"])
            if not pointmap_path.is_file():
                continue
            raw_xyz, conf = load_geometry(pointmap_path)
            height, width = chart_confidence[int(chart_index)].shape
            if conf.size != height * width:
                raise RuntimeError(
                    f"Raw/chart resolution mismatch for {pointmap_path}"
                )
            conf = conf.reshape(height, width)
            raw_xyz = raw_xyz.reshape(height, width, 3)
            aligned_xyz = (
                chart_points[int(chart_index)].astype(np.float32)
                / scale_factor
            )
            aligned_conf = chart_confidence[int(chart_index)]
            key = masks.source_name_for(image_path.name)
            channels = masks.masks[key]
            resized = [
                torch.nn.functional.interpolate(
                    value.detach().float()[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0]
                .bool()
                .numpy()
                for value in channels[:4]
            ]
            canopy = (
                resized[0]
                & resized[1]
                & ~resized[3]
                & np.isfinite(conf)
                & (conf >= 1.0)
            )
            canopy &= (
                np.isfinite(raw_xyz).all(axis=-1)
                & (np.linalg.norm(raw_xyz, axis=-1) > 1e-5)
            )
            candidates = np.flatnonzero(canopy)
            if not len(candidates):
                continue
            # Propagate only a small local MAtCha alignment correction from
            # the nearest rigid pixels.  A hard 25 cm cap prevents a flexible
            # facade chart from dragging non-rigid leaves across depth layers.
            aligned_valid = (
                np.isfinite(aligned_xyz).all(axis=-1)
                & (np.linalg.norm(aligned_xyz, axis=-1) > 1e-5)
                & np.isfinite(aligned_conf)
                & (aligned_conf > 0)
                & np.isfinite(raw_xyz).all(axis=-1)
            )
            corrected_xyz = raw_xyz
            if np.any(aligned_valid):
                distance, nearest = distance_transform_edt(
                    ~aligned_valid, return_indices=True
                )
                correction = aligned_xyz - raw_xyz
                local = correction[nearest[0], nearest[1]]
                global_correction = np.median(
                    correction[aligned_valid], axis=0
                )
                local = np.where(
                    (distance <= 48)[..., None],
                    local,
                    global_correction[None, None],
                )
                magnitude = np.linalg.norm(local, axis=-1, keepdims=True)
                local = local * np.minimum(
                    1.0, 0.25 / np.maximum(magnitude, 1e-8)
                )
                corrected_xyz = raw_xyz + local.astype(np.float32)
            # Confidence-only top-k collapses onto a few high-confidence
            # boughs and leaves most of the projected canopy transparent.
            # First retain the best point in each image-space cell, then use
            # the remaining confidence-ranked samples to add local detail.
            ranked = candidates[
                np.argsort(
                    conf.reshape(-1)[candidates], kind="stable"
                )[::-1]
            ]
            grid_side = max(
                1, int(np.ceil(np.sqrt(float(samples_per_view))))
            )
            rows, columns = np.divmod(ranked, width)
            cells = (
                (rows * grid_side // height) * grid_side
                + columns * grid_side // width
            )
            _, first = np.unique(cells, return_index=True)
            first = np.sort(first)
            coverage = ranked[first]
            if len(coverage) < int(samples_per_view):
                detail_mask = np.ones(len(ranked), dtype=bool)
                detail_mask[first] = False
                coverage = np.concatenate(
                    [
                        coverage,
                        ranked[detail_mask][
                            : int(samples_per_view) - len(coverage)
                        ],
                    ]
                )
            order = coverage[: int(samples_per_view)]
            if image_path.is_file():
                image = Image.open(image_path).convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            else:
                image = Image.open(
                    Path(store["dataset"])
                    / "images"
                    / images[image_id]["name"]
                ).convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            rgb = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
            for flat in order:
                value = corrected_xyz.reshape(-1, 3)[flat].astype(
                    np.float64
                )
                cell = (
                    int(image_id),
                    *tuple(
                        np.floor(value / float(voxel_size)).astype(int)
                    ),
                )
                if cell in occupied:
                    continue
                occupied.add(cell)
                result.append(
                    {
                        "id": -(len(result) + 1),
                        "xyz": value,
                        "rgb": rgb[flat],
                        "error": float(
                            (0.5 if active[int(chart_index)] else 1.0)
                            / max(
                                np.sqrt(float(conf.reshape(-1)[flat])),
                                0.25,
                            )
                        ),
                        "image_ids": np.asarray(
                            [image_id], dtype=np.int64
                        ),
                        "point2d_indices": np.asarray(
                            [-1], dtype=np.int64
                        ),
                        "tree_image_ids": np.asarray(
                            [image_id], dtype=np.int32
                        ),
                        "observation_camera_ids": np.asarray(
                            [image_id], dtype=np.int32
                        ),
                        "observation_uv": np.asarray(
                            [
                                [
                                    (int(flat) % width) / width,
                                    (int(flat) // width) / height,
                                ]
                            ],
                            dtype=np.float32,
                        ),
                        "observation_depth": np.asarray(
                            [
                                (
                                    value
                                    @ quaternion_to_rotation(
                                        images[image_id]["qvec"]
                                    ).T
                                    + np.asarray(
                                        images[image_id]["tvec"],
                                        dtype=np.float64,
                                    )
                                )[2]
                            ],
                            dtype=np.float32,
                        ),
                        "tree_sequence_count": 1,
                        "tree_fraction": (
                            1.0 if active[int(chart_index)] else 0.60
                        ),
                        "chart_sequence_local_sample": True,
                        "chart_alignment_active": bool(
                            active[int(chart_index)]
                        ),
                    }
                )
    if len(result) <= int(maximum_samples):
        return result
    # Cap only after every active Chart has contributed. An early global cap
    # made archive order decide the tree model and previously filled all
    # samples from the first traversal.
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for record in result:
        image_name = str(images[int(record["image_ids"][0])]["name"])
        sequence = image_name.split("__", 1)[0].split("/", 1)[0]
        by_sequence.setdefault(sequence, []).append(record)
    selected: list[dict[str, Any]] = []
    quota = max(1, int(maximum_samples) // len(by_sequence))
    remainder: list[dict[str, Any]] = []
    for sequence in sorted(by_sequence):
        rows = sorted(by_sequence[sequence], key=lambda row: row["error"])
        selected.extend(rows[:quota])
        remainder.extend(rows[quota:])
    if len(selected) < int(maximum_samples):
        selected.extend(
            sorted(remainder, key=lambda row: row["error"])[
                : int(maximum_samples) - len(selected)
            ]
        )
    return selected[: int(maximum_samples)]


def _dav2_foliage_samples(
    store: dict,
    images: dict[int, dict],
    cameras: dict[int, dict],
    masks: CambridgeMaskLookup,
    rigid_xyz: np.ndarray,
    *,
    samples_per_view: int = 128,
    maximum_samples: int = 120_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Back-project sparse per-view leaf owners from scaled DAV2 depth.

    A raw monocular depth map has an affine inverse-depth ambiguity.  We fit
    that ambiguity independently in every fixed Cambridge camera using only
    rigid-mask pixels covered by the MASt3R/MAtCha surface z-buffer.  Canopy
    pixels are then back-projected in the same camera, so a near tree cannot
    disappear merely because the small Chart subset omitted that timestamp.
    """
    index_path = artifact_path(store, "dav2_index", required=False)
    if index_path is None:
        return [], {"views": 0, "accepted_views": 0}
    records = json.loads(index_path.read_text(encoding="utf-8"))["records"]
    output: list[dict[str, Any]] = []
    accepted_views = 0
    rejected_views = 0
    rigid_xyz = np.asarray(rigid_xyz, dtype=np.float64)
    dataset = Path(store["dataset"])

    for image_id in sorted(images):
        image_record = images[image_id]
        stem = Path(str(image_record["name"])).stem
        depth_record = records.get(stem)
        if depth_record is None:
            continue
        depth_path = Path(depth_record["path"])
        if not depth_path.is_file():
            continue
        relative_depth = np.asarray(np.load(depth_path), dtype=np.float32)
        relative_depth = np.squeeze(relative_depth)
        if relative_depth.ndim != 2:
            rejected_views += 1
            continue
        source_height, source_width = relative_depth.shape
        scale = min(
            1.0,
            240.0 / max(source_width, 1),
            135.0 / max(source_height, 1),
        )
        if scale < 1.0:
            target_shape = (
                max(1, int(round(source_height * scale))),
                max(1, int(round(source_width * scale))),
            )
            relative_depth = (
                torch.nn.functional.interpolate(
                    torch.from_numpy(relative_depth)[None, None],
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                .numpy()
                .astype(np.float32)
            )
        height, width = relative_depth.shape
        camera = cameras[int(image_record["camera_id"])]
        if camera["model"] == "SIMPLE_PINHOLE":
            fx = fy = float(camera["params"][0])
            cx, cy = map(float, camera["params"][1:3])
        else:
            fx, fy, cx, cy = map(float, camera["params"][:4])
        sx = width / float(camera["width"])
        sy = height / float(camera["height"])
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        rotation = quaternion_to_rotation(image_record["qvec"])
        translation = np.asarray(
            image_record["tvec"], dtype=np.float64
        )
        camera_xyz = rigid_xyz @ rotation.T + translation[None]
        z = camera_xyz[:, 2]
        columns = np.rint(
            fx * camera_xyz[:, 0] / np.maximum(z, 1e-8) + cx
        ).astype(np.int64)
        rows = np.rint(
            fy * camera_xyz[:, 1] / np.maximum(z, 1e-8) + cy
        ).astype(np.int64)
        valid = (
            (z > 0.05)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        zbuffer = np.full(height * width, np.inf, dtype=np.float32)
        linear = rows[valid] * width + columns[valid]
        np.minimum.at(zbuffer, linear, z[valid].astype(np.float32))
        zbuffer = minimum_filter(
            zbuffer.reshape(height, width),
            size=5,
            mode="constant",
            cval=np.inf,
        )
        key = masks.source_name_for(image_record["name"])
        channels = masks.masks[key]
        resized = [
            torch.nn.functional.interpolate(
                value.detach().float()[None, None],
                size=(height, width),
                mode="nearest",
            )[0, 0]
            .bool()
            .numpy()
            for value in channels[:4]
        ]
        canopy = resized[0] & resized[1] & ~resized[3]
        rigid = resized[0] & resized[1] & resized[3]
        support = rigid & np.isfinite(zbuffer) & (zbuffer > 0)
        aligned, diagnostics = robust_align_inverse_depth(
            torch.from_numpy(
                1.0 / np.maximum(relative_depth, 1e-6)
            ),
            torch.from_numpy(zbuffer),
            torch.from_numpy(support),
            min_samples=64,
            max_samples=8_000,
            max_relative_rmse=0.35,
        )
        if not diagnostics.accepted:
            rejected_views += 1
            continue
        metric_depth = aligned.numpy()
        candidates = np.flatnonzero(
            canopy
            & np.isfinite(metric_depth)
            & (metric_depth > 0.05)
        )
        if not len(candidates):
            rejected_views += 1
            continue
        accepted_views += 1
        grid_side = max(
            1, int(np.ceil(np.sqrt(float(samples_per_view))))
        )
        candidate_rows, candidate_columns = np.divmod(
            candidates, width
        )
        cells = (
            (candidate_rows * grid_side // height) * grid_side
            + candidate_columns * grid_side // width
        )
        # Prefer central pixels in every cell; this is spatial coverage, not
        # a confidence hard gate.
        cell_center_x = (
            (cells % grid_side + 0.5) * width / grid_side
        )
        cell_center_y = (
            (cells // grid_side + 0.5) * height / grid_side
        )
        distance = (
            (candidate_columns - cell_center_x) ** 2
            + (candidate_rows - cell_center_y) ** 2
        )
        ranked = candidates[np.argsort(distance, kind="stable")]
        ranked_cells = cells[np.argsort(distance, kind="stable")]
        _, first = np.unique(ranked_cells, return_index=True)
        selected = ranked[np.sort(first)][: int(samples_per_view)]
        selected_rows, selected_columns = np.divmod(selected, width)
        selected_depth = metric_depth[
            selected_rows, selected_columns
        ].astype(np.float64)
        camera_points = np.stack(
            [
                (selected_columns - cx) * selected_depth / fx,
                (selected_rows - cy) * selected_depth / fy,
                selected_depth,
            ],
            axis=1,
        )
        world_points = (
            camera_points - translation[None]
        ) @ rotation
        image_path = dataset / "images" / image_record["name"]
        rgb_image = np.asarray(
            Image.open(image_path).convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
        footprint = np.clip(
            selected_depth
            * (min(height, width) / grid_side)
            / max(0.5 * (fx + fy), 1e-6)
            * 0.65,
            0.06,
            0.35,
        )
        for point, row, column, point_scale, depth_value in zip(
            world_points,
            selected_rows,
            selected_columns,
            footprint,
            selected_depth,
        ):
            output.append(
                {
                    "id": -(10_000_000 + len(output)),
                    "xyz": point.astype(np.float64),
                    "rgb": rgb_image[int(row), int(column)],
                    "error": float(
                        max(diagnostics.relative_rmse, 0.02)
                    ),
                    "image_ids": np.asarray(
                        [image_id], dtype=np.int64
                    ),
                    "point2d_indices": np.asarray(
                        [-1], dtype=np.int64
                    ),
                    "tree_image_ids": np.asarray(
                        [image_id], dtype=np.int32
                    ),
                    "observation_camera_ids": np.asarray(
                        [image_id], dtype=np.int32
                    ),
                    "observation_uv": np.asarray(
                        [
                            [
                                float(column) / width,
                                float(row) / height,
                            ]
                        ],
                        dtype=np.float32,
                    ),
                    "observation_depth": np.asarray(
                        [depth_value], dtype=np.float32
                    ),
                    "tree_sequence_count": 1,
                    "tree_fraction": float(
                        np.clip(
                            diagnostics.inlier_ratio, 0.55, 1.0
                        )
                    ),
                    "ray_footprint_scale": float(point_scale),
                    "dav2_sequence_local_sample": True,
                }
            )
        if len(output) >= int(maximum_samples):
            break
    return output[: int(maximum_samples)], {
        "views": int(accepted_views + rejected_views),
        "accepted_views": int(accepted_views),
        "rejected_views": int(rejected_views),
    }


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
    maximum_reprojection_error: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    xyz = np.stack([point["xyz"] for point in tracks]).astype(np.float32)
    rgb = (
        np.stack([point["rgb"] for point in tracks]).astype(np.float32)
        / 255.0
    )
    chart_local = np.asarray(
        [
            bool(point.get("chart_sequence_local_sample", False))
            for point in tracks
        ],
        dtype=bool,
    )
    dav2_local = np.asarray(
        [
            bool(point.get("dav2_sequence_local_sample", False))
            for point in tracks
        ],
        dtype=bool,
    )
    scales, quaternions, linearity = _track_frames(
        xyz, 0.008, 0.035, 0.10
    )
    ray_footprint = np.asarray(
        [point.get("ray_footprint_scale", 0.0) for point in tracks],
        dtype=np.float32,
    )
    scales[dav2_local] = np.maximum(
        scales[dav2_local],
        ray_footprint[dav2_local, None],
    )
    track_count = len(tracks)
    capacity = hull["support_camera_ids"].shape[1]
    support_camera_ids = np.full(
        (track_count, capacity), -1, dtype=np.int32
    )
    observation_camera_ids = np.full(
        (track_count, capacity), -1, dtype=np.int32
    )
    observation_uv = np.full(
        (track_count, capacity, 2), np.nan, dtype=np.float32
    )
    observation_depth = np.full(
        (track_count, capacity), np.nan, dtype=np.float32
    )
    for index, point in enumerate(tracks):
        values = np.unique(point["tree_image_ids"])[:capacity]
        support_camera_ids[index, : len(values)] = values
        cameras = np.asarray(
            point.get("observation_camera_ids", []), dtype=np.int32
        )[:capacity]
        uv = np.asarray(
            point.get("observation_uv", []), dtype=np.float32
        ).reshape(-1, 2)[:capacity]
        depth = np.asarray(
            point.get("observation_depth", []), dtype=np.float32
        ).reshape(-1)[:capacity]
        observation_count = min(len(cameras), len(uv), len(depth), capacity)
        observation_camera_ids[index, :observation_count] = cameras[
            :observation_count
        ]
        observation_uv[index, :observation_count] = uv[:observation_count]
        observation_depth[index, :observation_count] = depth[
            :observation_count
        ]
    error = np.asarray(
        [point["error"] for point in tracks], dtype=np.float32
    )
    covariance_scale = np.maximum(0.008, 0.012 + 0.01 * error)
    position_covariance = np.eye(3, dtype=np.float32)[None] * (
        covariance_scale[:, None, None] ** 2
    )
    sequence_support = np.asarray(
        [point["tree_sequence_count"] for point in tracks],
        dtype=np.int16,
    )
    # Cross-sequence linear tracks are stable trunk/branch anchors.  Tracks
    # supported only inside one traversal are useful high-frequency leaf
    # evidence, but must enter the sequence-conditioned branch rather than
    # contaminating canonical localization geometry.
    # PCA linearity is a proposal, not semantic trunk/branch evidence.
    # Until a cross-sequence line/cylinder inlier builder supplies that
    # evidence, emitting no skeleton is safer than inventing static floaters.
    trunk_or_branch_evidence = np.zeros(len(linearity), dtype=bool)
    layer_role = np.where(
        (sequence_support >= 2)
        & (linearity >= float(skeleton_linearity))
        & trunk_or_branch_evidence
        & (error <= maximum_reprojection_error),
        1,
        np.where(sequence_support < 2, 2, 0),
    ).astype(np.int8)
    # A single dense depth observation has no evidence for a canonical
    # trunk/branch identity. Even when its local neighbourhood looks linear,
    # it remains sequence-conditioned leaf evidence.
    layer_role[chart_local | dav2_local] = 2
    values = {
        "centers": xyz,
        "colors": rgb,
        "scales": scales,
        "opacities": np.where(
            dav2_local[:, None], 0.08, 0.025
        ).astype(np.float32),
        "quaternions": quaternions,
        "primitive_role": np.ones(track_count, dtype=np.int8),
        "layer_role": layer_role,
        "track_id": np.asarray(
            [point["id"] for point in tracks], dtype=np.int64
        ),
        "tree_instance_id": track_instances.astype(np.int32),
        "initialization_source": np.where(
            dav2_local,
            SOURCE_DAV2,
            np.where(chart_local, SOURCE_CHART, SOURCE_MAST3R),
        ).astype(np.int8),
        "occupancy_probability": np.asarray(
            [point["tree_fraction"] for point in tracks],
            dtype=np.float32,
        ),
        "position_covariance": position_covariance,
        "reprojection_error": error,
        "track_linearity": linearity,
        "support_camera_ids": support_camera_ids,
        "observation_camera_ids": observation_camera_ids,
        "observation_uv": observation_uv,
        "observation_depth": observation_depth,
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
            if key == "observation_camera_ids":
                hull[key] = np.full(
                    (len(hull["centers"]), capacity),
                    -1,
                    dtype=np.int32,
                )
            elif key == "observation_uv":
                hull[key] = np.full(
                    (len(hull["centers"]), capacity, 2),
                    np.nan,
                    dtype=np.float32,
                )
            elif key == "observation_depth":
                hull[key] = np.full(
                    (len(hull["centers"]), capacity),
                    np.nan,
                    dtype=np.float32,
                )
            else:
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
        "static_skeleton": int(
            (hull["layer_role"] == 1).sum() + (layer_role == 1).sum()
        ),
        "canonical_track_anchors": int((layer_role == 0).sum()),
        "canonical_crown": int((hull["layer_role"] == 0).sum()),
        "sequence_local_dynamic_tracks": int((layer_role == 2).sum()),
        "tree_instances": int(
            max(
                track_instances.max(),
                hull["tree_instance_id"].max(),
            )
        )
        + 1,
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
    masks = CambridgeMaskLookup(
        dataset, tree_mask, mask_indices=[0, 1, 2, 3]
    )
    mast3r_only = store.get("geometry_source") == "mast3r_only"
    if mast3r_only:
        tracks = _mast3r_tree_tracks(
            artifact_path(store, "mast3r_multiview_tracks"),
            images,
            # Keep sequence-local foliage observations.  Their layer role is
            # made dynamic below; canonical hull support still requires
            # cross-sequence agreement.
            minimum_track_observations=max(
                2, minimum_track_observations - 1
            ),
            maximum_reprojection_error=maximum_reprojection_error,
        )
        hull_tracks = _cross_view_tree_hull_anchors(
            tracks,
            images,
            cameras,
            masks,
            maximum_reprojection_error=maximum_reprojection_error,
        )
        cross_sequence_hull = bool(hull_tracks)
        if not hull_tracks:
            # Non-rigid foliage commonly has no valid cross-traversal
            # correspondence.  Build a traversal-local visual hull from true
            # observations, keep it out of localization, and let the dynamic
            # branch explain its high frequencies.
            hull_tracks = tracks
        native_track_count = len(tracks)
        chart_foliage = _chart_foliage_samples(
            store, images, masks
        )
        dav2_foliage, dav2_audit = _dav2_foliage_samples(
            store, images, cameras, masks, rigid_xyz
        )
        tracks = [*tracks, *chart_foliage, *dav2_foliage]
        # Sequence-local chart pointmaps are not correspondence tracks, but
        # they are valid ray/depth observations in their own fixed cameras.
        # Feed them to the probabilistic hull so canonical occupancy must be
        # corroborated by different traversals instead of being extruded from
        # whichever sequence happened to contain the surviving MASt3R tracks.
        hull_tracks = [*hull_tracks, *chart_foliage]
    else:
        points = read_points3d_binary_with_tracks(
            sparse / "points3D.bin"
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
        hull_tracks = tracks
        cross_sequence_hull = True
        native_track_count = len(tracks)
        chart_foliage = []
        dav2_foliage = []
        dav2_audit = {"views": 0, "accepted_views": 0}
    if not tracks:
        raise RuntimeError("No multi-view semantic tree tracks survived")
    track_xyz = np.stack([point["xyz"] for point in tracks])
    if mast3r_only and chart_foliage and native_track_count:
        native_xyz = track_xyz[:native_track_count]
        native_instances = cluster_tree_instances(native_xyz)
        chart_instances = native_instances[
            cKDTree(native_xyz).query(
                track_xyz[native_track_count:], k=1
            )[1]
        ]
        instances = np.concatenate(
            [native_instances, chart_instances]
        ).astype(np.int32)
    else:
        instances = cluster_tree_instances(track_xyz)
    view_records = [
        _camera_record(image, cameras[image["camera_id"]], masks)
        for image in images.values()
    ]
    if mast3r_only:
        observed_ids = {
            int(image_id)
            for track in hull_tracks
            for image_id in track.get("tree_image_ids", ())
        }
        observed_views = [
            view
            for view in view_records
            if int(view["image_id"]) in observed_ids
        ]
        selected = sequence_balanced_diverse_views(
            observed_views,
            limit=min(selected_view_count, len(observed_views)),
            minimum_center_distance=0.20,
        )
    else:
        selected = greedy_diverse_views(
            view_records,
            limit=selected_view_count,
            minimum_center_distance=0.75,
        )
    rigid_depth = sparse_rigid_depth_maps(selected, rigid_xyz)
    hull = build_instance_aware_canopy_volume(
        hull_tracks,
        images,
        cameras,
        masks,
        rigid_depth_maps=rigid_depth,
        selected_views=selected,
        minimum_support_sequences=2,
        voxel_size=voxel_size,
        maximum_voxels=maximum_voxels,
        seed=seed,
    )
    selected_views = hull.pop("selected_views")
    candidate_count = int(hull.pop("candidate_count"))
    rigid_depth_view_count = int(hull.pop("rigid_depth_view_count"))
    tree_instance_count = int(hull.pop("tree_instance_count"))
    ray_evidence = hull.pop("ray_evidence")
    merged, counts = _merge_foliage(
        hull,
        tracks,
        instances,
        skeleton_linearity=skeleton_linearity,
        maximum_reprojection_error=maximum_reprojection_error,
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
        "geometry_source": store.get("geometry_source", "legacy_mixed"),
        "colmap_points_or_tracks_used": not mast3r_only,
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
        "ray_evidence": {
            key: torch.from_numpy(value)
            for key, value in ray_evidence.items()
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
        "sequence_local_track_count": int(len(tracks)),
        "native_sequence_local_track_count": int(native_track_count),
        "chart_sequence_local_sample_count": int(
            len(chart_foliage)
        ),
        "dav2_sequence_local_sample_count": int(len(dav2_foliage)),
        "dav2_depth_alignment": dav2_audit,
        "cross_view_visual_hull_anchor_count": int(
            len(hull_tracks)
        ),
        "cross_sequence_visual_hull": bool(
            len({view["sequence_id"] for view in selected_views}) >= 2
        ),
        "localization_landmark_eligible": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
