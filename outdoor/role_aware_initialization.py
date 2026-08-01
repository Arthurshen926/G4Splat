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
from matcha.pointmap.calibration import retarget_pointmap_camera_rays
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
    evidence_conditioned_dynamic_opacity_ceiling,
    evidence_conditioned_leaf_optical_mass,
    quaternion_to_rotation,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary_with_tracks,
    semantic_tree_tracks,
)
from outdoor.foliage_view_graph import (
    greedy_diverse_views,
    instance_balanced_diverse_views,
    sequence_balanced_diverse_views,
    sequence_id,
)
from outdoor.scene_contract import sha256_file
from scripts.augment_foliage_seed_with_sfm_tracks import _track_frames


INITIALIZATION_VERSION = (
    "outdoor-role-aware-initialization-v61-camera-complete-strict-depth-"
    "posterior"
)
RIGID_CALIBRATED_INITIALIZATION_VERSION = (
    "outdoor-role-aware-initialization-v76-native-rigid-depth-continuous-"
    "posterior-exact-owner-2d-stratified-dual-bandwidth-calibrated-optical"
)
TEMPORAL_DAV2_AUGMENTATION_VERSION = (
    "outdoor-role-aware-initialization-v65-dense-evidence-preserving-exact-"
    "ray-depth-birth-spatial-coverage-strict-intervals-temporal-dav2-"
    "witnesses"
)

SOURCE_COLMAP = 0
SOURCE_MAST3R = 1
SOURCE_CHART = 2
# Foliage ``initialization_source`` predates surface renderer source roles.
# Keep its DAV2/dense-ray codes stable for existing foliage contracts, while
# assigning surface DAV2 births a code distinct from renderer-owned children.
SOURCE_DAV2 = 3
SOURCE_DENSE_RAY = 4
SOURCE_SURFACE_DAV2 = 4


def _effective_posterior_budget(
    score: np.ndarray,
    *,
    maximum_total: int,
) -> tuple[int, float]:
    """Return an evidence-sized capacity without a binary quality gate."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    score = np.where(np.isfinite(score), np.maximum(score, 0.0), 0.0)
    maximum_total = max(int(maximum_total), 0)
    posterior_mass = float(score.sum())
    posterior_square_mass = float(np.square(score).sum())
    if (
        maximum_total == 0
        or posterior_mass <= 0.0
        or posterior_square_mass <= 1e-12
    ):
        return 0, 0.0
    effective_count = (
        posterior_mass * posterior_mass
        / posterior_square_mass
    )
    return (
        min(maximum_total, int(np.ceil(effective_count))),
        float(effective_count),
    )


def _chart_hypothesis_authority(
    independent_support: np.ndarray,
    consensus_confirmed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate renderer coverage from permanent geometry authority.

    MAtCha's complete aligned raster remains a permanent source-resolution
    *observation* in the Evidence Store.  It is not necessary, or safe, to
    instantiate every single-view pixel as a simultaneously composited
    Gaussian: many mutually inconsistent, low-opacity sheets still accumulate
    substantial optical mass.  Renderer births therefore require independent
    metric/cross-Chart support, while only stricter cross-view consensus makes
    a row a persistent geometry observation.  Unsupported pixels are not
    deleted from the Chart factor; they simply do not acquire primitive
    ownership before rendered residual/topology evidence asks for capacity.
    """
    independent_support = np.asarray(independent_support, dtype=bool)
    consensus_confirmed = np.asarray(consensus_confirmed, dtype=bool)
    if independent_support.shape != consensus_confirmed.shape:
        raise ValueError(
            "Chart independent-support and consensus arrays must match"
        )
    if bool((consensus_confirmed & ~independent_support).any()):
        raise ValueError(
            "Cross-view confirmed Chart rows lack independent support"
        )
    renderer_admitted = independent_support.copy()
    geometry_persistent = consensus_confirmed.copy()
    initial_opacity = np.where(
        geometry_persistent, 0.05, 0.025
    ).astype(np.float32)
    return renderer_admitted, geometry_persistent, initial_opacity


def _dav2_rigid_hole_completion_samples(
    store: dict,
    *,
    dataset: Path,
    rgb_root: Path,
    tree_mask_pickle: Path,
    rigid_xyz: np.ndarray,
    selected_view_count: int,
    samples_per_view: int,
    maximum_total: int,
    cross_sequence_radius: float,
) -> dict[str, Any]:
    """Create low-authority rigid births only where the metric seed is empty.

    DAV2 is never treated as metric geometry.  Each selected real camera first
    fits its affine inverse-depth ambiguity to the rendered-coordinate
    MASt3R/MAtCha scaffold.  Samples are proposed only in rigid semantic
    pixels not covered by that scaffold, then ranked continuously by
    cross-traversal agreement, alignment quality and measured footprint
    deficit.  Poor proposals therefore start nearly transparent and are
    pruned normally instead of becoming a second opaque monocular sheet.
    """
    empty = {
        "xyz": np.empty((0, 3), dtype=np.float32),
        "rgb": np.empty((0, 3), dtype=np.float32),
        "sigma": np.empty(0, dtype=np.float32),
        "confidence": np.empty(0, dtype=np.float32),
        "scales": np.empty((0, 2), dtype=np.float32),
        "quaternions": np.empty((0, 4), dtype=np.float32),
        "normals": np.empty((0, 3), dtype=np.float32),
        "initial_opacity": np.empty(0, dtype=np.float32),
        "observation_view_name": np.empty(0, dtype="<U1"),
        "observation_uv": np.empty((0, 2), dtype=np.float32),
        "observation_depth": np.empty(0, dtype=np.float32),
        "audit": {
            "enabled": False,
            "reason": "no_dav2_or_nonpositive_budget",
            "selected": 0,
        },
    }
    selected_view_count = max(int(selected_view_count), 0)
    samples_per_view = max(int(samples_per_view), 0)
    maximum_total = max(int(maximum_total), 0)
    if not selected_view_count or not samples_per_view or not maximum_total:
        return empty
    index_path = artifact_path(store, "dav2_index", required=False)
    if index_path is None:
        return empty
    records = json.loads(index_path.read_text(encoding="utf-8")).get(
        "records", {}
    )
    if not records:
        return empty
    rigid_xyz = np.asarray(rigid_xyz, dtype=np.float64).reshape(-1, 3)
    if len(rigid_xyz) < 3:
        return empty
    sparse = Path(dataset) / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    masks = CambridgeMaskLookup(
        dataset, tree_mask_pickle, mask_indices=[0, 1, 2, 3]
    )
    view_records = [
        _camera_record(image, cameras[image["camera_id"]], masks)
        for image in images.values()
        if Path(str(image["name"])).stem in records
    ]
    # The shared view selector is sequence/pose aware.  Reverse its canopy
    # preference for this rigid-only pass without changing camera identity.
    rigid_view_records = [
        {
            **row,
            "canopy_fraction": 1.0
            - float(np.clip(row["canopy_fraction"], 0.0, 1.0)),
        }
        for row in view_records
    ]
    selected_views = sequence_balanced_diverse_views(
        rigid_view_records,
        limit=min(selected_view_count, len(rigid_view_records)),
        minimum_center_distance=0.20,
    )
    selected_ids = {int(row["image_id"]) for row in selected_views}

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    sequence_parts: list[np.ndarray] = []
    view_parts: list[np.ndarray] = []
    footprint_parts: list[np.ndarray] = []
    alignment_parts: list[np.ndarray] = []
    camera_center_parts: list[np.ndarray] = []
    ray_direction_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    quaternion_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    observation_view_name_parts: list[np.ndarray] = []
    observation_uv_parts: list[np.ndarray] = []
    observation_depth_parts: list[np.ndarray] = []
    accepted_views = 0
    rejected_alignment_views = 0
    proposed_pixels = 0
    for image_id in sorted(selected_ids):
        image_record = images[int(image_id)]
        stem = Path(str(image_record["name"])).stem
        depth_record = records.get(stem)
        if depth_record is None:
            continue
        depth_path = Path(depth_record["path"])
        if not depth_path.is_file():
            continue
        relative_depth = np.squeeze(
            np.asarray(np.load(depth_path), dtype=np.float32)
        )
        if relative_depth.ndim != 2:
            rejected_alignment_views += 1
            continue
        source_height, source_width = relative_depth.shape
        resize_scale = min(
            1.0,
            240.0 / max(source_width, 1),
            135.0 / max(source_height, 1),
        )
        if resize_scale < 1.0:
            relative_depth = (
                torch.nn.functional.interpolate(
                    torch.from_numpy(relative_depth)[None, None],
                    size=(
                        max(1, int(round(source_height * resize_scale))),
                        max(1, int(round(source_width * resize_scale))),
                    ),
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
        depth = camera_xyz[:, 2]
        columns = np.rint(
            fx * camera_xyz[:, 0] / np.maximum(depth, 1e-8) + cx
        ).astype(np.int64)
        rows = np.rint(
            fy * camera_xyz[:, 1] / np.maximum(depth, 1e-8) + cy
        ).astype(np.int64)
        valid_projection = (
            (depth > 0.05)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        zbuffer = np.full(height * width, np.inf, dtype=np.float32)
        linear = (
            rows[valid_projection] * width + columns[valid_projection]
        )
        np.minimum.at(
            zbuffer,
            linear,
            depth[valid_projection].astype(np.float32),
        )
        zbuffer = minimum_filter(
            zbuffer.reshape(height, width),
            size=5,
            mode="constant",
            cval=np.inf,
        )
        key = masks.source_name_for(image_record["name"])
        channels = masks.masks[key]
        resized_masks = [
            torch.nn.functional.interpolate(
                value.detach().float()[None, None],
                size=(height, width),
                mode="nearest",
            )[0, 0]
            .bool()
            .numpy()
            for value in channels[:4]
        ]
        rigid = np.logical_and.reduce(resized_masks)
        covered = np.isfinite(zbuffer) & (zbuffer > 0)
        support = rigid & covered
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
            rejected_alignment_views += 1
            continue
        metric_depth = aligned.numpy()
        holes = (
            rigid
            & ~covered
            & np.isfinite(metric_depth)
            & (metric_depth > 0.05)
        )
        proposed_pixels += int(holes.sum())
        if not bool(holes.any()):
            continue
        hole_distance = distance_transform_edt(~covered).astype(
            np.float32
        )
        selected = _uniform_confidence_samples(
            holes,
            hole_distance,
            samples_per_view,
        )
        if not len(selected):
            continue
        accepted_views += 1
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
        grid_rows, grid_columns = np.indices((height, width))
        full_camera_points = np.stack(
            [
                (grid_columns - cx) * metric_depth / fx,
                (grid_rows - cy) * metric_depth / fy,
                metric_depth,
            ],
            axis=-1,
        )
        full_world_points = (
            full_camera_points.reshape(-1, 3) - translation[None]
        ) @ rotation
        full_world_points = full_world_points.reshape(height, width, 3)
        local_scales, local_quaternions, local_normals = (
            _pointmap_surface_frames(
                full_world_points,
                selected_rows,
                selected_columns,
            )
        )
        # A DAV2 birth is an image-space surface observation.  Its tangent
        # frame must therefore come from the aligned native depth Jacobian,
        # not a global 3-D kNN that can mix the foreground church, a rear
        # facade and adjacent depth layers.  Expand the one-pixel tangent only
        # by the measured spacing of retained samples in this source view.
        local_spacing = _sampled_point_footprint_multiplier(
            selected_rows,
            selected_columns,
        )
        local_scales = np.clip(
            local_scales * local_spacing[:, None],
            0.004,
            0.18,
        ).astype(np.float32)
        camera_center = (-translation[None]) @ rotation
        ray_direction = world_points - camera_center
        ray_direction /= np.maximum(
            np.linalg.norm(ray_direction, axis=1, keepdims=True), 1e-12
        )
        image_name = Path(str(image_record["name"])).name
        image_path = Path(rgb_root) / image_name
        if not image_path.is_file():
            image_path = Path(rgb_root) / str(image_record["name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image_handle:
            rgb_image = np.asarray(
                image_handle.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        rgb = (
            rgb_image[selected_rows, selected_columns] / 255.0
        ).astype(np.float32)
        nominal_spacing = np.sqrt(
            max(float(holes.sum()), 1.0)
            / max(float(len(selected)), 1.0)
        )
        footprint = np.clip(
            selected_depth
            * nominal_spacing
            / max(0.5 * (fx + fy), 1e-6),
            0.015,
            0.18,
        ).astype(np.float32)
        alignment_quality = float(
            np.clip(
                diagnostics.inlier_ratio
                / (1.0 + diagnostics.relative_rmse),
                0.0,
                1.0,
            )
        )
        xyz_parts.append(world_points.astype(np.float32))
        rgb_parts.append(rgb)
        sequence_parts.append(
            np.full(
                len(selected),
                str(sequence_id(stem)),
                dtype=f"<U{max(len(str(sequence_id(stem))), 1)}",
            )
        )
        view_parts.append(
            np.full(len(selected), int(image_id), dtype=np.int32)
        )
        footprint_parts.append(footprint)
        alignment_parts.append(
            np.full(
                len(selected), alignment_quality, dtype=np.float32
            )
        )
        camera_center_parts.append(
            np.broadcast_to(
                camera_center.astype(np.float32),
                world_points.shape,
            ).copy()
        )
        ray_direction_parts.append(ray_direction.astype(np.float32))
        scale_parts.append(local_scales)
        quaternion_parts.append(local_quaternions.astype(np.float32))
        normal_parts.append(local_normals.astype(np.float32))
        observation_view_name_parts.append(
            np.full(len(selected), stem, dtype=f"<U{max(len(stem), 1)}")
        )
        observation_uv_parts.append(
            np.column_stack(
                [
                    (selected_columns.astype(np.float32) + 0.5)
                    / max(float(width), 1.0),
                    (selected_rows.astype(np.float32) + 0.5)
                    / max(float(height), 1.0),
                ]
            ).astype(np.float32)
        )
        observation_depth_parts.append(
            selected_depth.astype(np.float32)
        )
    if not xyz_parts:
        result = dict(empty)
        result["audit"] = {
            "enabled": True,
            "selected_view_count": len(selected_views),
            "accepted_view_count": accepted_views,
            "rejected_alignment_view_count": rejected_alignment_views,
            "proposed_rigid_hole_pixels": proposed_pixels,
            "selected": 0,
            "reason": "no_metrically_aligned_rigid_hole_candidates",
        }
        return result

    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    sequence = np.concatenate(sequence_parts)
    view_id = np.concatenate(view_parts)
    footprint = np.concatenate(footprint_parts)
    alignment_quality = np.concatenate(alignment_parts)
    camera_center = np.concatenate(camera_center_parts)
    ray_direction = np.concatenate(ray_direction_parts)
    native_scales = np.concatenate(scale_parts)
    native_quaternions = np.concatenate(quaternion_parts)
    native_normals = np.concatenate(normal_parts)
    observation_view_name = np.concatenate(
        observation_view_name_parts
    )
    observation_uv = np.concatenate(observation_uv_parts)
    observation_depth = np.concatenate(observation_depth_parts)
    peer_distance, peer_index = _different_sequence_nearest(
        xyz, sequence
    )
    xyz, cross_ray_posterior, cross_ray_audit = (
        _cross_sequence_ray_posterior(
            xyz,
            camera_center,
            ray_direction,
            peer_index,
        )
    )
    nearest_existing = cKDTree(rigid_xyz).query(
        xyz, k=1, workers=-1
    )[0].astype(np.float32)
    radius = max(float(cross_sequence_radius), 1e-4)
    peer_score = np.exp(
        -0.5 * np.square(peer_distance / radius)
    ).astype(np.float32)
    peer_score[~np.isfinite(peer_distance)] = 0.0
    normalized_deficit = np.clip(
        (nearest_existing - footprint)
        / np.maximum(footprint + 0.02, 1e-4),
        0.0,
        1.0,
    ).astype(np.float32)
    score = (
        alignment_quality
        * peer_score
        * (0.25 + 0.75 * cross_ray_posterior)
        * (0.10 + 0.90 * normalized_deficit)
    )
    envelope_keep, envelope_audit = _rigid_scene_envelope_mask(
        xyz, rigid_xyz
    )
    candidate_indices = np.flatnonzero(envelope_keep)
    # Compact duplicate proposals globally while retaining the strongest
    # continuous posterior in every small world-space cell.
    order = candidate_indices[
        np.argsort(score[candidate_indices], kind="stable")[::-1]
    ]
    voxel = np.floor(xyz[order] / 0.03).astype(np.int64)
    _, first = np.unique(voxel, axis=0, return_index=True)
    unique_indices = order[np.sort(first)]
    unique_score = score[unique_indices]
    # A mathematically zero cross-sequence posterior is absence of evidence,
    # not a low-quality observation. Excluding exact zeros keeps the
    # view-balancing allocator from spending an evidence-sized budget on
    # unsupported rows; no positive quality threshold is introduced.
    # Values below float32 machine epsilon cannot change the persisted
    # confidence or opacity from their zero-posterior bases. Treating them as
    # renderer evidence would preserve only an underflow artefact, not add a
    # hand-tuned geometric quality threshold.
    positive = unique_score > np.finfo(np.float32).eps
    zero_posterior_candidate_count = int((~positive).sum())
    unique_indices = unique_indices[positive]
    unique_score = unique_score[positive]
    _, effective_posterior_count = _effective_posterior_budget(
        unique_score,
        maximum_total=maximum_total,
    )
    capacity_budget = min(maximum_total, len(unique_indices))
    if capacity_budget == 0:
        result = dict(empty)
        result["audit"] = {
            "enabled": True,
            "selected_view_count": len(selected_views),
            "accepted_view_count": accepted_views,
            "rejected_alignment_view_count": rejected_alignment_views,
            "candidate_count": int(len(xyz)),
            "voxel_unique_candidate_count": int(len(unique_indices)),
            "zero_posterior_candidate_count": (
                zero_posterior_candidate_count
            ),
            "selected": 0,
            "reason": "zero_cross_sequence_posterior_mass",
            "hard_quality_gate": False,
        }
        return result
    # Every representable positive posterior may compete for renderer
    # capacity.  Confidence and opacity remain continuous below; using the
    # posterior effective sample size as a row-count gate previously reduced
    # 56k unique proposals to 16k and recreated the measured coverage hole.
    selected = _balanced_pointmap_seed_cap(
        unique_indices,
        budget=capacity_budget,
        view_id=view_id,
        score=score,
    )
    if len(selected) < 3:
        result = dict(empty)
        result["audit"] = {
            "enabled": True,
            "selected_view_count": len(selected_views),
            "accepted_view_count": accepted_views,
            "rejected_alignment_view_count": rejected_alignment_views,
            "candidate_count": int(len(xyz)),
            "selected": 0,
            "reason": "fewer_than_three_bounded_candidates",
        }
        return result
    selected_xyz = xyz[selected].astype(np.float32)
    selected_peer_distance = peer_distance[selected]
    selected_score = np.clip(score[selected], 0.0, 1.0)
    sigma = np.clip(
        0.02
        + 0.25
        * np.minimum(
            np.nan_to_num(
                selected_peer_distance,
                nan=radius * 2.0,
                posinf=radius * 2.0,
            ),
            radius * 2.0,
        ),
        0.02,
        0.35,
    ).astype(np.float32)
    scales = native_scales[selected].astype(np.float32)
    quaternions = native_quaternions[selected].astype(np.float32)
    normals = native_normals[selected].astype(np.float32)
    scales = np.clip(
        np.maximum(
            scales,
            0.50 * footprint[selected, None],
        ),
        0.004,
        0.18,
    ).astype(np.float32)
    confidence = (
        0.05 + 0.95 * selected_score
    ).astype(np.float32)
    # The numerical base sits just above the trainer's 0.005 cull while an
    # unseen camera is waiting to be sampled.  All visual authority above
    # that survival floor is posterior-proportional.
    initial_opacity = (
        0.0051 + 0.0449 * selected_score
    ).astype(np.float32)
    finite_peer = selected_peer_distance[
        np.isfinite(selected_peer_distance)
    ]
    audit = {
        "enabled": True,
        "selection_contract": (
            "mast3r_matcha_metric_scaffold_calibrates_dav2__"
            "rigid_hole_deficit_times_cross_sequence_continuous_posterior"
        ),
        "selected_view_count": int(len(selected_views)),
        "accepted_view_count": int(accepted_views),
        "rejected_alignment_view_count": int(
            rejected_alignment_views
        ),
        "proposed_rigid_hole_pixels": int(proposed_pixels),
        "candidate_count": int(len(xyz)),
        "bounded_candidate_count": int(envelope_keep.sum()),
        "voxel_unique_candidate_count": int(len(unique_indices)),
        "zero_posterior_candidate_count": (
            zero_posterior_candidate_count
        ),
        "selected": int(len(selected)),
        "maximum_total": int(maximum_total),
        "effective_posterior_count": float(effective_posterior_count),
        "effective_capacity_budget": int(capacity_budget),
        "capacity_policy": (
            "all_float32_representable_positive_posterior_rows_up_to_budget"
        ),
        "samples_per_view": int(samples_per_view),
        "cross_sequence_radius": float(radius),
        "cross_sequence_distance_median": (
            float(np.median(finite_peer)) if len(finite_peer) else None
        ),
        "selection_score_percentiles": np.percentile(
            selected_score, [0, 10, 50, 90, 100]
        ).tolist(),
        "scene_envelope": envelope_audit,
        "cross_sequence_ray_posterior": cross_ray_audit,
        "surface_frame_policy": (
            "source_aligned_dav2_depth_jacobian_times_retained_pixel_spacing"
        ),
        "permanent_metric_authority": False,
        "initial_opacity_policy": "0.0051_plus_0.0449_times_score",
        "hard_quality_gate": False,
    }
    return {
        "xyz": selected_xyz,
        "rgb": rgb[selected].astype(np.float32),
        "sigma": sigma,
        "confidence": confidence,
        "scales": scales,
        "quaternions": quaternions,
        "normals": normals,
        "initial_opacity": initial_opacity,
        "observation_view_name": observation_view_name[selected],
        "observation_uv": observation_uv[selected],
        "observation_depth": observation_depth[selected],
        "audit": audit,
    }


def _rigid_scene_envelope_mask(
    candidates: np.ndarray,
    rigid_reference: np.ndarray,
    *,
    reference_quantile: float = 0.999,
    expansion: float = 1.35,
    minimum_margin: float = 5.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Reject only catastrophic depth rays, never ordinary novel coverage."""
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    rigid_reference = np.asarray(
        rigid_reference, dtype=np.float64
    ).reshape(-1, 3)
    if not len(rigid_reference):
        return np.ones(len(candidates), dtype=bool), {
            "reference_radius": 0.0,
            "maximum_radius": float("inf"),
            "rejected": 0,
        }
    if not 0.0 < float(reference_quantile) < 1.0:
        raise ValueError("Scene-envelope quantile must lie in (0, 1)")
    center = np.median(rigid_reference, axis=0)
    reference_radius = float(
        np.quantile(
            np.linalg.norm(
                rigid_reference - center[None], axis=1
            ),
            float(reference_quantile),
        )
    )
    maximum_radius = max(
        reference_radius * float(expansion),
        reference_radius + float(minimum_margin),
    )
    candidate_radius = np.linalg.norm(
        candidates - center[None], axis=1
    )
    keep = (
        np.isfinite(candidates).all(axis=1)
        & np.isfinite(candidate_radius)
        & (candidate_radius <= maximum_radius)
    )
    return keep, {
        "reference_radius": reference_radius,
        "maximum_radius": maximum_radius,
        "rejected": int((~keep).sum()),
    }


def _load_mast3r_pointmap_geometry(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode world points/confidence without boxing the large RGB payload."""
    path = Path(path)
    raw = path.read_bytes()
    point_marker = b'"points":'
    confidence_marker = b'"confs":'
    point_start = raw.find(point_marker)
    confidence_start = raw.find(confidence_marker)
    if point_start < 0 or confidence_start < 0:
        raise RuntimeError(f"Malformed MASt3R pointmap: {path}")
    point_start += len(point_marker)
    confidence_value_start = confidence_start + len(confidence_marker)
    payload_end = raw.rfind(b"}")
    point_payload = raw[point_start:confidence_start].rstrip()
    if point_payload.endswith(b","):
        point_payload = point_payload[:-1].rstrip()
    bracket_table = bytes.maketrans(b"[]", b"  ")
    point_values = np.fromstring(
        point_payload.translate(bracket_table),
        dtype=np.float32,
        sep=",",
    )
    confidence_values = np.fromstring(
        raw[confidence_value_start:payload_end]
        .strip()
        .translate(bracket_table),
        dtype=np.float32,
        sep=",",
    )
    pixel_count = int(confidence_values.size)
    if pixel_count <= 0 or point_values.size != 3 * pixel_count:
        raise RuntimeError(
            f"Point/conf size mismatch in MASt3R pointmap {path}: "
            f"points={point_values.size}, confidence={pixel_count}"
        )
    return point_values.reshape(pixel_count, 3), confidence_values


def _load_pointmap_cross_sequence_posterior(
    record: dict[str, Any],
    native_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    """Load the immutable per-pixel metric posterior for one pointmap.

    The posterior builder evaluates each native MASt3R pixel against
    independently traversed Cambridge sequences.  Recomputing support after
    subsampling the pointmaps changes both the reference density and the
    support semantics, so initialization must consume the same posterior as
    the permanent training factor.
    """
    posterior_record = record.get("cross_sequence_posterior")
    if posterior_record is None:
        return None
    path = Path(str(posterior_record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = posterior_record.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(
        expected_bytes
    ):
        raise RuntimeError(
            f"Indexed MASt3R posterior changed size: {path}"
        )
    expected_sha256 = posterior_record.get("sha256")
    if expected_sha256 is not None and sha256_file(path) != str(
        expected_sha256
    ):
        raise RuntimeError(
            f"Indexed MASt3R posterior changed content: {path}"
        )
    with np.load(path, allow_pickle=False) as posterior:
        schema = str(posterior["schema_version"].item())
        if schema != "mast3r-cross-sequence-pointmap-posterior-v1":
            raise RuntimeError(
                f"Unsupported MASt3R pointmap posterior {schema!r}"
            )
        stored_shape = tuple(
            map(int, posterior["native_shape"].reshape(-1))
        )
        if stored_shape != tuple(map(int, native_shape)):
            raise RuntimeError(
                "MASt3R pointmap posterior native shape changed: "
                f"{stored_shape} != {native_shape}"
            )
        precision = posterior["precision"].astype(np.float32)
        supported = posterior["cross_sequence_supported"].astype(bool)
        distance = posterior["cross_sequence_distance"].astype(np.float32)
    for name, value in (
        ("precision", precision),
        ("supported", supported),
        ("distance", distance),
    ):
        if value.shape != native_shape:
            raise RuntimeError(
                f"MASt3R posterior {name} shape {value.shape} does not "
                f"match pointmap {native_shape}"
            )
    precision = np.nan_to_num(
        precision, nan=0.0, posinf=0.0, neginf=0.0
    )
    precision = np.clip(precision, 0.0, 1.0)
    supported &= precision > 0
    distance = np.where(
        np.isfinite(distance) & (distance >= 0), distance, np.inf
    ).astype(np.float32)
    return {
        "precision": precision,
        "supported": supported,
        "distance": distance,
    }


def _native_pointmap_shape(
    pixel_count: int,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Recover the native raster shape using the source image aspect ratio."""
    pixel_count = int(pixel_count)
    image_width, image_height = map(int, image_size)
    if pixel_count <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("Pointmap size and source image size must be positive")
    target_aspect = image_width / image_height
    candidates = []
    for height in range(1, int(np.sqrt(pixel_count)) + 1):
        if pixel_count % height:
            continue
        width = pixel_count // height
        for candidate_height, candidate_width in (
            (height, width),
            (width, height),
        ):
            aspect_error = abs(
                np.log(
                    max(candidate_width / candidate_height, 1e-8)
                    / target_aspect
                )
            )
            candidates.append(
                (aspect_error, candidate_height, candidate_width)
            )
    if not candidates:
        raise RuntimeError(
            f"Cannot factor MASt3R pointmap pixel count {pixel_count}"
        )
    _, height, width = min(candidates)
    return int(height), int(width)


def _pointmap_fixed_camera_validity(
    points: np.ndarray,
    native_shape: tuple[int, int],
    scene_record: dict,
    *,
    maximum_reprojection_error_px: float = 0.75,
) -> np.ndarray:
    """Prove that every retained 3-D point measures its own camera pixel."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    height, width = map(int, native_shape)
    if len(points) != height * width:
        raise ValueError("Pointmap/native raster shape mismatch")
    world_to_camera = np.asarray(
        scene_record["T_world_to_camera"], dtype=np.float64
    )
    camera = (
        points @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    depth = camera[:, 2]
    contract = scene_record["camera"]
    fx, fy, cx, cy = _pointmap_fixed_intrinsics(
        native_shape, contract
    )
    projected_u = fx * camera[:, 0] / np.maximum(depth, 1e-8) + cx
    projected_v = fy * camera[:, 1] / np.maximum(depth, 1e-8) + cy
    rows, columns = np.divmod(np.arange(len(points)), width)
    reprojection = np.hypot(
        projected_u - columns,
        projected_v - rows,
    )
    return (
        np.isfinite(camera).all(axis=1)
        & (depth > 0.05)
        & np.isfinite(reprojection)
        & (reprojection <= float(maximum_reprojection_error_px))
    )


def _pointmap_fixed_intrinsics(
    native_shape: tuple[int, int],
    camera_contract: dict,
) -> np.ndarray:
    height, width = map(int, native_shape)
    source_width = int(camera_contract["width"])
    source_height = int(camera_contract["height"])
    scale_x = width / max(source_width, 1)
    scale_y = height / max(source_height, 1)
    return np.asarray(
        [
            float(camera_contract["fx"]) * scale_x,
            float(camera_contract["fy"]) * scale_y,
            float(camera_contract["cx"]) * scale_x,
            float(camera_contract["cy"]) * scale_y,
        ],
        dtype=np.float64,
    )


def _retarget_pointmap_fixed_camera_rays(
    points: np.ndarray,
    native_shape: tuple[int, int],
    scene_record: dict,
) -> np.ndarray:
    """Retain camera-z while moving each sample onto the exact fixed-K ray."""
    height, width = map(int, native_shape)
    pointmap = np.asarray(points).reshape(height, width, 3)
    return retarget_pointmap_camera_rays(
        pointmap,
        np.asarray(scene_record["T_camera_to_world"], dtype=np.float64),
        _pointmap_fixed_intrinsics(
            native_shape, scene_record["camera"]
        ),
    ).reshape(-1, 3)


def _pointmap_surface_frames(
    pointmap: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build perspective-aware tangent frames from adjacent pointmap pixels."""
    height, width = pointmap.shape[:2]
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
    finite = np.isfinite(frames).all(axis=(1, 2))
    frames[~finite] = np.eye(3, dtype=np.float32)
    normal[~finite] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
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
    footprint = np.clip(footprint, 0.003, 0.08).astype(np.float32)
    return footprint, quaternion.astype(np.float32), normal.astype(np.float32)


def _sampled_point_footprint_multiplier(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    maximum: float = 4.0,
) -> np.ndarray:
    """Match a pointmap surfel footprint to the retained pixel spacing.

    ``_pointmap_surface_frames`` measures a two-pixel finite difference and
    therefore produces an approximately one-pixel Gaussian scale.  Keeping
    only a uniform subset of a dense pointmap without compensating that scale
    leaves deterministic screen-space holes (for 6k of 512x288 pixels, the
    retained spacing is normally about five pixels).  Use the actual nearest
    retained observation distance rather than a fixed world-space expansion.

    The factor is ``spacing / 2`` because the native tangent spans the two
    adjacent pixels.  Dense or locally detailed samples never shrink, while a
    conservative cap prevents isolated samples from becoming giant surfels.
    """
    rows = np.asarray(rows, dtype=np.float32).reshape(-1)
    columns = np.asarray(columns, dtype=np.float32).reshape(-1)
    if len(rows) != len(columns):
        raise ValueError("sample rows and columns must have the same length")
    if len(rows) < 2:
        return np.ones(len(rows), dtype=np.float32)
    pixels = np.column_stack([columns, rows])
    distance = cKDTree(pixels).query(pixels, k=2, workers=-1)[0][:, 1]
    distance = np.where(np.isfinite(distance), distance, 2.0)
    return np.clip(
        0.5 * distance, 1.0, max(float(maximum), 1.0)
    ).astype(np.float32)


def _different_sequence_nearest_distance(
    xyz: np.ndarray,
    sequence: np.ndarray,
    *,
    maximum_neighbours: int = 64,
) -> np.ndarray:
    """Nearest spatial support from an independently traversed sequence."""
    distance, _ = _different_sequence_nearest(
        xyz,
        sequence,
        maximum_neighbours=maximum_neighbours,
    )
    return distance


def _different_sequence_nearest(
    xyz: np.ndarray,
    sequence: np.ndarray,
    *,
    maximum_neighbours: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest distance and row from an independently traversed sequence."""
    xyz = np.asarray(xyz, dtype=np.float32)
    sequence = np.asarray(sequence)
    if len(xyz) != len(sequence):
        raise ValueError("XYZ and sequence arrays must have the same length")
    if not len(xyz):
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    neighbour_count = min(max(int(maximum_neighbours), 2), len(xyz))
    distance, indices = cKDTree(xyz).query(
        xyz, k=neighbour_count, workers=-1
    )
    distance = np.asarray(distance)
    indices = np.asarray(indices)
    if distance.ndim == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    result = np.full(len(xyz), np.inf, dtype=np.float32)
    result_index = np.full(len(xyz), -1, dtype=np.int64)
    for rank in range(1, neighbour_count):
        take = np.isinf(result) & (
            sequence[indices[:, rank]] != sequence
        )
        result[take] = distance[take, rank]
        result_index[take] = indices[take, rank]
    return result, result_index


def _cross_sequence_ray_posterior(
    xyz: np.ndarray,
    camera_center: np.ndarray,
    ray_direction: np.ndarray,
    peer_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse paired monocular proposals using their calibrated world rays.

    A nearest 3D proposal is only a correspondence hypothesis.  The closest
    points between the two exact camera rays provide the actual geometric
    posterior: ray gap, triangulation angle and consistency with both
    monocular depth observations all contribute continuously.  No quality
    threshold turns a hypothesis into metric authority.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    camera_center = np.asarray(
        camera_center, dtype=np.float64
    ).reshape(-1, 3)
    ray_direction = np.asarray(
        ray_direction, dtype=np.float64
    ).reshape(-1, 3)
    peer_index = np.asarray(peer_index, dtype=np.int64).reshape(-1)
    if not (
        len(xyz)
        == len(camera_center)
        == len(ray_direction)
        == len(peer_index)
    ):
        raise ValueError("Cross-ray posterior arrays must have equal rows")
    count = len(xyz)
    fused = xyz.copy()
    posterior = np.zeros(count, dtype=np.float64)
    ray_gap = np.full(count, np.inf, dtype=np.float64)
    angle = np.zeros(count, dtype=np.float64)
    valid_peer = (peer_index >= 0) & (peer_index < count)
    rows = np.flatnonzero(valid_peer)
    if not len(rows):
        return (
            fused.astype(np.float32),
            posterior.astype(np.float32),
            {
                "paired": 0,
                "finite_triangulation": 0,
                "ray_gap_percentiles": [],
                "angle_degree_percentiles": [],
                "posterior_percentiles": [],
            },
        )
    peer = peer_index[rows]
    first_direction = ray_direction[rows]
    second_direction = ray_direction[peer]
    first_direction /= np.maximum(
        np.linalg.norm(first_direction, axis=1, keepdims=True), 1e-12
    )
    second_direction /= np.maximum(
        np.linalg.norm(second_direction, axis=1, keepdims=True), 1e-12
    )
    center_delta = camera_center[rows] - camera_center[peer]
    cosine = np.sum(first_direction * second_direction, axis=1)
    denominator = np.maximum(1.0 - cosine * cosine, 0.0)
    first_dot = np.sum(first_direction * center_delta, axis=1)
    second_dot = np.sum(second_direction * center_delta, axis=1)
    safe_denominator = np.maximum(denominator, 1e-12)
    first_depth = (
        cosine * second_dot - first_dot
    ) / safe_denominator
    second_depth = (
        second_dot - cosine * first_dot
    ) / safe_denominator
    first_closest = (
        camera_center[rows]
        + first_depth[:, None] * first_direction
    )
    second_closest = (
        camera_center[peer]
        + second_depth[:, None] * second_direction
    )
    midpoint = 0.5 * (first_closest + second_closest)
    current_gap = np.linalg.norm(first_closest - second_closest, axis=1)
    current_angle = np.sqrt(denominator)
    first_prior_depth = np.sum(
        (xyz[rows] - camera_center[rows]) * first_direction, axis=1
    )
    second_prior_depth = np.sum(
        (xyz[peer] - camera_center[peer]) * second_direction, axis=1
    )
    first_sigma = 0.10 + 0.03 * np.maximum(first_prior_depth, 0.0)
    second_sigma = 0.10 + 0.03 * np.maximum(second_prior_depth, 0.0)
    finite = (
        np.isfinite(midpoint).all(axis=1)
        & np.isfinite(current_gap)
        & (denominator > 1e-8)
        & (first_depth > 0.0)
        & (second_depth > 0.0)
    )
    current_posterior = (
        current_angle / (current_angle + 0.05)
        * np.exp(-0.5 * np.square(current_gap / 0.08))
        * np.exp(
            -0.5
            * np.square(
                (first_depth - first_prior_depth)
                / np.maximum(first_sigma, 1e-6)
            )
        )
        * np.exp(
            -0.5
            * np.square(
                (second_depth - second_prior_depth)
                / np.maximum(second_sigma, 1e-6)
            )
        )
    )
    current_posterior[~finite] = 0.0
    # A weak intersection remains a weak monocular proposal.  A strong pair
    # converges to the midpoint of the two calibrated rays.  This continuous
    # update avoids producing two displaced opaque sheets from the pair.
    fused[rows] = (
        (1.0 - current_posterior[:, None]) * xyz[rows]
        + current_posterior[:, None] * midpoint
    )
    posterior[rows] = current_posterior
    ray_gap[rows] = current_gap
    angle[rows] = np.degrees(np.arcsin(np.clip(current_angle, 0.0, 1.0)))
    finite_rows = rows[finite]
    return (
        fused.astype(np.float32),
        posterior.astype(np.float32),
        {
            "paired": int(len(rows)),
            "finite_triangulation": int(len(finite_rows)),
            "ray_gap_percentiles": (
                np.percentile(ray_gap[finite_rows], [0, 10, 50, 90, 100])
                .astype(float)
                .tolist()
                if len(finite_rows)
                else []
            ),
            "angle_degree_percentiles": (
                np.percentile(angle[finite_rows], [0, 10, 50, 90, 100])
                .astype(float)
                .tolist()
                if len(finite_rows)
                else []
            ),
            "posterior_percentiles": (
                np.percentile(posterior[rows], [0, 10, 50, 90, 100])
                .astype(float)
                .tolist()
                if len(rows)
                else []
            ),
        },
    )


def _balanced_pointmap_seed_cap(
    indices: np.ndarray,
    *,
    budget: int,
    view_id: np.ndarray,
    score: np.ndarray,
) -> np.ndarray:
    """Keep a score-prioritized, view-balanced subset of pointmap rows."""
    indices = np.asarray(indices, dtype=np.int64)
    budget = min(max(int(budget), 0), len(indices))
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    if budget == len(indices):
        return indices
    views = np.unique(view_id[indices])
    per_view_cap = max(int(np.ceil(budget / max(len(views), 1))), 1)
    balanced = []
    for current_view in views:
        candidates = indices[view_id[indices] == current_view]
        ranked = candidates[
            np.argsort(score[candidates], kind="stable")[::-1]
        ]
        balanced.extend(ranked[:per_view_cap].tolist())
    selected = np.asarray(balanced, dtype=np.int64)
    if len(selected) > budget:
        selected = selected[
            np.argsort(score[selected], kind="stable")[-budget:]
        ]
    elif len(selected) < budget:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        fill = remaining[
            np.argsort(score[remaining], kind="stable")[
                -min(budget - len(selected), len(remaining)) :
            ]
        ]
        selected = np.concatenate([selected, fill])
    return selected


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


def _uniform_confidence_samples(
    valid: np.ndarray,
    confidence: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Retain image-wide coverage before spending budget on local detail."""
    valid = np.asarray(valid, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float32)
    height, width = valid.shape
    candidates = np.flatnonzero(valid)
    limit = min(max(int(maximum), 0), len(candidates))
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    rows, columns = np.divmod(candidates, width)
    aspect = max(width / max(height, 1), 1e-6)
    grid_x = max(1, int(np.ceil(np.sqrt(limit * aspect))))
    grid_y = max(1, int(np.ceil(limit / grid_x)))
    cells = (
        np.minimum(rows * grid_y // max(height, 1), grid_y - 1)
        * grid_x
        + np.minimum(
            columns * grid_x // max(width, 1), grid_x - 1
        )
    )
    confidence_flat = confidence.reshape(-1)
    ranked = np.argsort(confidence_flat[candidates], kind="stable")[::-1]
    occupied = np.zeros(grid_x * grid_y, dtype=bool)
    coverage = []
    coverage_candidate_indices = []
    for candidate_index in ranked:
        cell = int(cells[candidate_index])
        if occupied[cell]:
            continue
        occupied[cell] = True
        coverage.append(int(candidates[candidate_index]))
        coverage_candidate_indices.append(int(candidate_index))
        if len(coverage) == limit:
            break
    remaining = limit - len(coverage)
    if remaining:
        available = np.ones(len(candidates), dtype=bool)
        available[np.asarray(coverage_candidate_indices, dtype=np.int64)] = False
        detail = candidates[available]
        detail = detail[
            np.argsort(confidence_flat[detail], kind="stable")[-remaining:]
        ]
        coverage.extend(map(int, detail))
    return np.asarray(coverage, dtype=np.int64)


def _sample_dense_mast3r_rigid_seeds(
    store: dict,
    *,
    dataset: Path,
    tree_mask_pickle: Path,
    samples_per_view: int,
    maximum_total: int,
    minimum_confidence: float,
    cross_sequence_radius: float,
    voxel_size: float,
    maximum_single_sequence_fraction: float,
) -> dict[str, np.ndarray]:
    """Sample the dense fixed-world MASt3R pointmaps as rigid 2D surfels.

    The old MASt3R-only path reduced roughly eight million real pointmap
    observations to fewer than two thousand source-star tracks, then asked
    low-resolution Charts to cover the whole church.  This function retains a
    bounded, image-uniform subset of the original metric observations.  A
    different traversal raises confidence, but lack of overlap is uncertainty
    rather than a hard deletion gate.
    """
    index_path = artifact_path(
        store, "mast3r_pointmap_index", required=False
    )
    if index_path is None:
        return {
            "xyz": np.empty((0, 3), dtype=np.float32),
            "rgb": np.empty((0, 3), dtype=np.float32),
            "sigma": np.empty(0, dtype=np.float32),
            "confidence": np.empty(0, dtype=np.float32),
            "view_id": np.empty(0, dtype=np.int32),
            "uv": np.empty((0, 2), dtype=np.float32),
            "scales": np.empty((0, 2), dtype=np.float32),
            "sampling_footprint_multiplier": np.empty(
                0, dtype=np.float32
            ),
            "quaternions": np.empty((0, 4), dtype=np.float32),
            "normals": np.empty((0, 3), dtype=np.float32),
            "cross_sequence_supported": np.empty(0, dtype=bool),
            "cross_sequence_distance": np.empty(0, dtype=np.float32),
            "cross_sequence_precision": np.empty(0, dtype=np.float32),
        }
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("coordinate_frame") != "cambridge_fixed_world":
        raise RuntimeError(
            "Dense MASt3R pointmaps are not in the fixed Cambridge world"
        )
    records = payload.get("records", {})
    camera_order = payload.get("camera_order", list(records))
    if set(camera_order) != set(records):
        raise RuntimeError("MASt3R pointmap index camera order is incomplete")
    lookup = CambridgeMaskLookup(
        dataset, tree_mask_pickle, mask_indices=[0, 1, 2, 3]
    )
    scene_payload = json.loads(
        Path(store["scene_contract"]).read_text(encoding="utf-8")
    )
    scene_records = {
        Path(str(row["image_name"])).stem: row
        for row in scene_payload["records"]
    }
    missing_scene_cameras = sorted(set(camera_order) - set(scene_records))
    if missing_scene_cameras:
        raise RuntimeError(
            "Pointmap cameras are missing from the immutable scene contract: "
            + ", ".join(missing_scene_cameras[:5])
        )
    xyz_parts = []
    rgb_parts = []
    confidence_parts = []
    view_parts = []
    sequence_parts = []
    uv_parts = []
    scale_parts = []
    sampling_multiplier_parts = []
    quaternion_parts = []
    normal_parts = []
    posterior_available_parts = []
    posterior_precision_parts = []
    posterior_supported_parts = []
    posterior_distance_parts = []
    for view_id, stem in enumerate(camera_order):
        record = records[stem]
        pointmap_path = Path(record["path"])
        if (
            not pointmap_path.is_file()
            or pointmap_path.stat().st_size != int(record["bytes"])
        ):
            raise RuntimeError(
                f"Indexed MASt3R pointmap changed or disappeared: "
                f"{pointmap_path}"
            )
        staged_name = lookup.staged_by_stem.get(stem, f"{stem}.png")
        image_path = dataset / "images" / staged_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image_handle:
            source_size = image_handle.size
        point_flat, confidence_flat = _load_mast3r_pointmap_geometry(
            pointmap_path
        )
        height, width = _native_pointmap_shape(
            len(confidence_flat), source_size
        )
        point_flat = _retarget_pointmap_fixed_camera_rays(
            point_flat,
            (height, width),
            scene_records[stem],
        )
        pointmap = point_flat.reshape(height, width, 3)
        confidence = confidence_flat.reshape(height, width)
        fixed_camera_valid = _pointmap_fixed_camera_validity(
            point_flat,
            (height, width),
            scene_records[stem],
        ).reshape(height, width)
        posterior = _load_pointmap_cross_sequence_posterior(
            record, (height, width)
        )
        masks = lookup.get_index_masks(
            staged_name,
            (0, 1, 2, 3),
            (height, width),
            torch.device("cpu"),
        ).numpy()
        rigid = masks.all(axis=0)
        valid = (
            rigid
            & np.isfinite(confidence)
            & (confidence >= float(minimum_confidence))
            & np.isfinite(pointmap).all(axis=-1)
            & (np.linalg.norm(pointmap, axis=-1) > 1e-5)
            & fixed_camera_valid
        )
        selected = _uniform_confidence_samples(
            valid, confidence, samples_per_view
        )
        if not len(selected):
            continue
        rows, columns = np.divmod(selected, width)
        xyz = pointmap.reshape(-1, 3)[selected].astype(np.float32)
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            rgb = (
                np.asarray(image, dtype=np.float32)
                .reshape(-1, 3)[selected]
                / 255.0
            )
        scales, quaternions, normals = _pointmap_surface_frames(
            pointmap, rows, columns
        )
        sampling_multiplier = _sampled_point_footprint_multiplier(
            rows, columns
        )
        # Preserve broad coverage at initialization without imposing a
        # world-space blur: the multiplier is derived in the native
        # observation image and applied along the measured tangent axes.
        scales = np.clip(
            scales * sampling_multiplier[:, None], 0.003, 0.18
        ).astype(np.float32)
        xyz_parts.append(xyz)
        rgb_parts.append(rgb.astype(np.float32))
        confidence_parts.append(
            confidence.reshape(-1)[selected].astype(np.float32)
        )
        view_parts.append(
            np.full(len(selected), int(view_id), dtype=np.int32)
        )
        sequence_parts.append(
            np.full(
                len(selected),
                str(sequence_id(stem)),
                dtype=f"<U{max(len(str(sequence_id(stem))), 1)}",
            )
        )
        uv_parts.append(
            np.column_stack(
                [
                    (columns + 0.5) / float(width),
                    (rows + 0.5) / float(height),
                ]
            ).astype(np.float32)
        )
        scale_parts.append(scales)
        sampling_multiplier_parts.append(sampling_multiplier)
        quaternion_parts.append(quaternions)
        normal_parts.append(normals)
        posterior_available_parts.append(
            np.full(len(selected), posterior is not None, dtype=bool)
        )
        if posterior is None:
            posterior_precision_parts.append(
                np.zeros(len(selected), dtype=np.float32)
            )
            posterior_supported_parts.append(
                np.zeros(len(selected), dtype=bool)
            )
            posterior_distance_parts.append(
                np.full(len(selected), np.inf, dtype=np.float32)
            )
        else:
            posterior_precision_parts.append(
                posterior["precision"].reshape(-1)[selected]
            )
            posterior_supported_parts.append(
                posterior["supported"].reshape(-1)[selected]
            )
            posterior_distance_parts.append(
                posterior["distance"].reshape(-1)[selected]
            )
    if not xyz_parts:
        raise RuntimeError(
            "MASt3R pointmap index contains no valid rigid observations"
        )
    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    raw_confidence = np.concatenate(confidence_parts)
    view_id = np.concatenate(view_parts)
    sequence = np.concatenate(sequence_parts)
    uv = np.concatenate(uv_parts)
    scales = np.concatenate(scale_parts)
    sampling_multiplier = np.concatenate(sampling_multiplier_parts)
    quaternions = np.concatenate(quaternion_parts)
    normals = np.concatenate(normal_parts)
    posterior_available = np.concatenate(posterior_available_parts)
    posterior_precision = np.concatenate(posterior_precision_parts)
    cross_supported = np.concatenate(posterior_supported_parts)
    cross_distance = np.concatenate(posterior_distance_parts)
    if not bool(posterior_available.all()):
        fallback_distance = _different_sequence_nearest_distance(
            xyz, sequence
        )
        fallback_rows = ~posterior_available
        cross_distance[fallback_rows] = fallback_distance[fallback_rows]
        cross_supported[fallback_rows] = (
            fallback_distance[fallback_rows]
            <= float(cross_sequence_radius)
        )
        posterior_precision[fallback_rows] = np.where(
            cross_supported[fallback_rows], 1.0, 0.30
        )
    posterior_precision = np.clip(
        posterior_precision, 0.0, 1.0
    ).astype(np.float32)

    # Deduplicate metric points globally, preferring independent traversal
    # support and then MASt3R confidence.  Per-view UV sampling has already
    # protected broad image coverage before this spatial compaction.
    normalized_confidence = np.empty_like(raw_confidence)
    for current_view in np.unique(view_id):
        rows = view_id == current_view
        median = max(float(np.median(raw_confidence[rows])), 1e-6)
        normalized_confidence[rows] = raw_confidence[rows] / median
    score = (
        cross_supported.astype(np.float32) * 16.0
        + np.clip(
            normalized_confidence
            * np.maximum(posterior_precision, 0.03),
            0.0,
            8.0,
        )
    )
    order = np.argsort(score, kind="stable")[::-1]
    voxel = np.floor(
        xyz[order] / max(float(voxel_size), 1e-5)
    ).astype(np.int64)
    _, first = np.unique(voxel, axis=0, return_index=True)
    keep = order[np.sort(first)]
    # Independent traversals define the metric surface posterior.  A large
    # single-sequence tail is still useful for broad bootstrap coverage, but
    # allowing it to fill the seed budget lets locally plausible yet globally
    # displaced pointmaps consume the adaptive densification capacity.  Cap
    # that tail explicitly while retaining it as low-precision training
    # evidence in the immutable pointmap factor.
    maximum_total = max(int(maximum_total), 0)
    maximum_single_sequence_fraction = float(
        np.clip(maximum_single_sequence_fraction, 0.0, 1.0)
    )
    cross_candidates = keep[cross_supported[keep]]
    single_candidates = keep[~cross_supported[keep]]
    cross_keep = _balanced_pointmap_seed_cap(
        cross_candidates,
        budget=maximum_total,
        view_id=view_id,
        score=score,
    )
    remaining = max(maximum_total - len(cross_keep), 0)
    single_budget = min(
        remaining,
        int(np.floor(maximum_total * maximum_single_sequence_fraction)),
    )
    single_keep = _balanced_pointmap_seed_cap(
        single_candidates,
        budget=single_budget,
        view_id=view_id,
        score=score,
    )
    keep = np.concatenate([cross_keep, single_keep])
    confidence = np.clip(normalized_confidence[keep], 0.1, 4.0)
    confidence *= np.where(
        cross_supported[keep],
        0.35 + 0.65 * posterior_precision[keep],
        0.25 + 0.75 * posterior_precision[keep],
    )
    finite_cross = np.where(
        np.isfinite(cross_distance[keep]),
        cross_distance[keep],
        4.0 * float(cross_sequence_radius),
    )
    sigma = np.clip(
        0.012 / np.sqrt(np.maximum(confidence, 0.05))
        + 0.08
        * np.clip(
            finite_cross / max(float(cross_sequence_radius), 1e-5),
            0.0,
            4.0,
        ),
        0.008,
        0.35,
    ).astype(np.float32)
    return {
        "xyz": xyz[keep].astype(np.float32),
        "rgb": rgb[keep].astype(np.float32),
        "sigma": sigma,
        "confidence": confidence.astype(np.float32),
        "view_id": view_id[keep].astype(np.int32),
        "uv": uv[keep].astype(np.float32),
        "scales": scales[keep].astype(np.float32),
        "sampling_footprint_multiplier": sampling_multiplier[keep].astype(
            np.float32
        ),
        "quaternions": quaternions[keep].astype(np.float32),
        "normals": normals[keep].astype(np.float32),
        "cross_sequence_supported": cross_supported[keep],
        "cross_sequence_distance": cross_distance[keep].astype(np.float32),
        "cross_sequence_precision": posterior_precision[keep].astype(
            np.float32
        ),
    }


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
    chart_consensus: Path | None,
    scene_contract: Path,
    dataset: Path,
    tree_mask_pickle: Path,
    rigid_reference: np.ndarray,
    per_chart: int,
    maximum_total: int,
    consensus_radius: float,
    seed: int,
) -> dict[str, Any]:
    lookup = CambridgeMaskLookup(
        dataset, tree_mask_pickle, mask_indices=[0, 1, 2, 3]
    )
    camera_payload = json.loads(chart_cameras.read_text(encoding="utf-8"))
    files = [Path(value) for value in camera_payload["filepaths"]]
    scene_payload = json.loads(
        Path(scene_contract).read_text(encoding="utf-8")
    )
    scene_records = {
        Path(str(row["image_name"])).stem: row
        for row in scene_payload["records"]
    }
    missing_scene_cameras = sorted(
        {path.stem for path in files} - set(scene_records)
    )
    if missing_scene_cameras:
        raise RuntimeError(
            "Chart cameras are missing from the immutable scene contract: "
            + ", ".join(missing_scene_cameras[:5])
        )
    consensus = None
    if chart_consensus is not None:
        with np.load(chart_consensus, allow_pickle=False) as sidecar:
            required = {
                "depths",
                "support_counts",
                "correction_mask",
                "consistency_weights",
                "image_names",
            }
            missing = required - set(sidecar.files)
            if missing:
                raise RuntimeError(
                    "Chart consensus lacks required arrays: "
                    + ", ".join(sorted(missing))
                )
            sidecar_names = [
                Path(str(value)).stem for value in sidecar["image_names"]
            ]
            if sidecar_names != [path.stem for path in files]:
                raise RuntimeError(
                    "Chart consensus camera order does not match Chart atlas"
                )
            minimum_support = int(
                sidecar["minimum_consensus_views"].item()
                if "minimum_consensus_views" in sidecar
                else 2
            )
            consensus = {
                "depths": sidecar["depths"].astype(np.float32),
                "support": sidecar["support_counts"].astype(np.uint8),
                "accepted": sidecar["correction_mask"].astype(bool),
                "weight": sidecar["consistency_weights"].astype(np.float32),
                "minimum_support": minimum_support,
            }
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
    selected_consensus_confirmed = []
    selected_sampling_multiplier = []
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
        # MASt3R tracks and MAtCha Charts are complementary evidence.  The
        # former validates the metric frame; it must not be used as a spatial
        # acceptance mask for the latter.  Doing so collapsed a continuous
        # facade atlas to the small neighbourhood of sparse SfM tracks.
        reference_tree = cKDTree(rigid_reference) if len(rigid_reference) else None
        for chart_index in np.nonzero(active)[0]:
            scene_record = scene_records[files[chart_index].stem]
            pointmap = _retarget_pointmap_fixed_camera_rays(
                points[chart_index],
                confidence[chart_index].shape,
                scene_record,
            ).reshape((*confidence[chart_index].shape, 3))
            raw_pointmap = pointmap.copy()
            conf = confidence[chart_index]
            height, width = conf.shape
            consensus_confirmed = np.zeros((height, width), dtype=bool)
            consensus_contradicted = np.zeros((height, width), dtype=bool)
            consensus_trust = np.full(
                (height, width), 0.15, dtype=np.float32
            )
            if consensus is not None:
                support = consensus["support"][chart_index]
                supported = support >= int(consensus["minimum_support"])
                consensus_confirmed = (
                    supported & consensus["accepted"][chart_index]
                )
                consensus_contradicted = supported & ~consensus_confirmed
                corrected_depth = (
                    consensus["depths"][chart_index] / scale_factor
                )
                rows_full, columns_full = np.mgrid[0:height, 0:width]
                fx, fy, cx, cy = _pointmap_fixed_intrinsics(
                    (height, width), scene_record["camera"]
                )
                camera = np.stack(
                    [
                        (columns_full - cx) * corrected_depth / fx,
                        (rows_full - cy) * corrected_depth / fy,
                        corrected_depth,
                    ],
                    axis=-1,
                )
                transform = np.asarray(
                    scene_record["T_camera_to_world"],
                    dtype=np.float64,
                )
                corrected_xyz = (
                    camera @ transform[:3, :3].T
                    + transform[:3, 3]
                ).astype(np.float32)
                # Chart-to-Chart agreement is not independent evidence: two
                # deformed sheets can reinforce the same wrong facade depth.
                # Accept a correction only when a cross-traversal MASt3R
                # reference supports it and it does not move farther from
                # that reference than the raw MAtCha Chart.
                independently_supported = np.zeros(
                    (height, width), dtype=bool
                )
                proposed = np.flatnonzero(consensus_confirmed)
                if reference_tree is not None and len(proposed):
                    raw_distance = reference_tree.query(
                        raw_pointmap.reshape(-1, 3)[proposed], k=1
                    )[0]
                    corrected_distance = reference_tree.query(
                        corrected_xyz.reshape(-1, 3)[proposed], k=1
                    )[0]
                    accepted_independent = (
                        (corrected_distance <= 0.15)
                        & (corrected_distance <= raw_distance + 0.02)
                    )
                    independently_supported.reshape(-1)[
                        proposed[accepted_independent]
                    ] = True
                consensus_confirmed &= independently_supported
                pointmap[consensus_confirmed] = corrected_xyz[
                    consensus_confirmed
                ]
                # A Chart consensus candidate rejected only for lack of
                # independent overlap remains a low-trust raw bootstrap
                # observation.  It is not falsely labelled a contradiction.
                consensus_contradicted = (
                    supported & ~consensus["accepted"][chart_index]
                )
                consensus_trust[consensus_confirmed] = np.clip(
                    consensus["weight"][chart_index][consensus_confirmed],
                    0.10,
                    1.0,
                )
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
                & ~consensus_contradicted
                & np.isfinite(pointmap).all(axis=-1)
                & (np.linalg.norm(pointmap, axis=-1) > 1e-5)
                & np.isfinite(conf)
                & (conf > 0)
            )
            flat = np.flatnonzero(valid)
            if not len(flat):
                continue
            limit = min(int(per_chart), len(flat))
            # Spend most of the budget on UV coverage: take the strongest
            # valid sample in each approximately uniform image cell.  The
            # remaining budget is confidence-ranked detail.  A global
            # strongest/random mixture left large textureless wall regions
            # without any structural primitive.
            aspect = max(width / max(height, 1), 1e-6)
            grid_x = max(1, int(np.ceil(np.sqrt(limit * aspect))))
            grid_y = max(1, int(np.ceil(limit / grid_x)))
            rows_all, columns_all = np.unravel_index(flat, (height, width))
            cells = (
                np.minimum(rows_all * grid_y // max(height, 1), grid_y - 1)
                * grid_x
                + np.minimum(
                    columns_all * grid_x // max(width, 1), grid_x - 1
                )
            )
            confidence_flat = conf.reshape(-1)
            order = np.argsort(confidence_flat[flat])[::-1]
            occupied = np.zeros(grid_x * grid_y, dtype=bool)
            coverage = []
            for candidate in order:
                cell = int(cells[candidate])
                if not occupied[cell]:
                    occupied[cell] = True
                    coverage.append(int(flat[candidate]))
                    if len(coverage) == limit:
                        break
            coverage = np.asarray(coverage, dtype=np.int64)
            remaining = limit - len(coverage)
            if remaining:
                pool = np.setdiff1d(flat, coverage, assume_unique=False)
                detail = pool[
                    np.argsort(confidence_flat[pool])[-remaining:]
                ]
                chosen = np.concatenate([coverage, detail])
            else:
                chosen = coverage
            xyz = pointmap.reshape(-1, 3)[chosen].astype(np.float32)
            conf_value = conf.reshape(-1)[chosen].astype(np.float32)
            trust_value = consensus_trust.reshape(-1)[chosen]
            confirmed_value = consensus_confirmed.reshape(-1)[chosen]
            cross_distance = None
            if reference_tree is not None:
                cross_distance = reference_tree.query(xyz, k=1)[0]
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
            base_confidence = conf[valid]
            normalized = (
                conf_value
                / max(float(np.median(base_confidence)), 1e-6)
                * trust_value
            )
            sigma = np.clip(
                (0.025 / scale_factor) / np.sqrt(normalized),
                0.012,
                0.35,
            )
            # Distance to MASt3R is retained as uncertainty, not a hard gate.
            # Unsupported Chart regions therefore remain available to cover
            # facades while moving more conservatively during optimization.
            if cross_distance is not None:
                support_ratio = np.clip(
                    cross_distance / max(float(consensus_radius), 1e-4),
                    0.0,
                    6.0,
                )
                sigma = np.maximum(
                    sigma,
                    0.012 * (1.0 + 0.35 * support_ratio),
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
            sampling_multiplier = _sampled_point_footprint_multiplier(
                rows, columns
            )
            # Confirmed Chart pixels may cover their actual retained UV-cell
            # spacing.  Unsupported pixels are only low-opacity coverage
            # witnesses, so broadening them by the full sparse-sampling
            # multiplier would turn an uncertain sheet into a paint layer.
            trust_multiplier = 1.0 + (
                sampling_multiplier - 1.0
            ) * np.clip(trust_value, 0.15, 1.0)
            footprint = np.clip(
                footprint * trust_multiplier[:, None],
                0.003,
                0.12,
            ).astype(np.float32)
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
            selected_consensus_confirmed.append(confirmed_value)
            selected_sampling_multiplier.append(
                trust_multiplier.astype(np.float32)
            )
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
            "consensus_confirmed": np.empty(0, dtype=bool),
            "independent_support": np.empty(0, dtype=bool),
            "renderer_admitted": np.empty(0, dtype=bool),
            "geometry_persistent": np.empty(0, dtype=bool),
            "initial_opacity": np.empty(0, dtype=np.float32),
            "scene_envelope": {
                "reference_radius": 0.0,
                "maximum_radius": float("inf"),
                "rejected": 0,
            },
            "sampling_footprint_multiplier": np.empty(
                0, dtype=np.float32
            ),
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
    consensus_confirmed = np.concatenate(selected_consensus_confirmed)
    sampling_multiplier = np.concatenate(selected_sampling_multiplier)
    envelope_keep, scene_envelope = _rigid_scene_envelope_mask(
        xyz,
        rigid_reference,
    )
    (
        xyz,
        rgb,
        sigma,
        confidence,
        chart_id,
        uv,
        scales,
        quaternions,
        normals,
        consensus_confirmed,
        sampling_multiplier,
    ) = (
        xyz[envelope_keep],
        rgb[envelope_keep],
        sigma[envelope_keep],
        confidence[envelope_keep],
        chart_id[envelope_keep],
        uv[envelope_keep],
        scales[envelope_keep],
        quaternions[envelope_keep],
        normals[envelope_keep],
        consensus_confirmed[envelope_keep],
        sampling_multiplier[envelope_keep],
    )
    independent_support = np.zeros(len(xyz), dtype=bool)
    # Measure independent support without turning it into a renderer-birth
    # gate. MAtCha is the only dense no-COLMAP facade coverage producer in
    # this front end. Deleting every single-Chart sample outside a small
    # MASt3R/cross-Chart radius collapsed roughly 156k uniformly sampled
    # observations to about 32k anchors, after which RGB densification had to
    # invent missing surfaces from screen gradients.
    #
    # Unsupported rows remain explicitly non-persistent, start at opacity
    # 0.01, and enter ordinary prune/replace after bootstrap. They are
    # coverage hypotheses, never geometry truth. The support bit is retained
    # for provenance and uncertainty audits.
    if len(xyz):
        metric_distance = (
            reference_tree.query(xyz, k=1)[0]
            if reference_tree is not None
            else np.full(len(xyz), np.inf, dtype=np.float32)
        )
        neighbour_count = min(64, len(xyz))
        if neighbour_count > 1:
            neighbour_distance, neighbour_index = cKDTree(xyz).query(
                xyz, k=neighbour_count
            )
            neighbour_distance = np.atleast_2d(neighbour_distance)
            neighbour_index = np.atleast_2d(neighbour_index)
            cross_chart_distance = np.full(
                len(xyz), np.inf, dtype=np.float32
            )
            cross_chart_index = np.full(len(xyz), -1, dtype=np.int64)
            for rank in range(1, neighbour_count):
                different_chart = (
                    chart_id[neighbour_index[:, rank]] != chart_id
                )
                unresolved = ~np.isfinite(cross_chart_distance)
                take = different_chart & unresolved
                cross_chart_distance[take] = neighbour_distance[take, rank]
                cross_chart_index[take] = neighbour_index[take, rank]
        else:
            cross_chart_distance = np.full(
                len(xyz), np.inf, dtype=np.float32
            )
            cross_chart_index = np.full(len(xyz), -1, dtype=np.int64)
        normal_agreement = np.zeros(len(xyz), dtype=bool)
        paired = cross_chart_index >= 0
        if bool(paired.any()):
            normal_agreement[paired] = (
                np.abs(
                    np.sum(
                        normals[paired]
                        * normals[cross_chart_index[paired]],
                        axis=1,
                    )
                )
                >= 0.75
            )
        metric_limit = max(0.40, 4.0 * float(consensus_radius))
        cross_chart_limit = max(0.15, 1.5 * float(consensus_radius))
        independent_support = (
            (metric_distance <= metric_limit)
            | (
                (cross_chart_distance <= cross_chart_limit)
                & normal_agreement
            )
        )
    if len(xyz) > int(maximum_total):
        order = np.argsort(confidence)[-int(maximum_total) :]
        (
            xyz,
            rgb,
            sigma,
            confidence,
            chart_id,
            uv,
            scales,
            quaternions,
            normals,
            consensus_confirmed,
            independent_support,
            sampling_multiplier,
        ) = (
            xyz[order],
            rgb[order],
            sigma[order],
            confidence[order],
            chart_id[order],
            uv[order],
            scales[order],
            quaternions[order],
            normals[order],
            consensus_confirmed[order],
            independent_support[order],
            sampling_multiplier[order],
        )
    (
        renderer_admitted,
        geometry_persistent,
        initial_opacity,
    ) = _chart_hypothesis_authority(
        independent_support,
        consensus_confirmed,
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
        "consensus_confirmed": consensus_confirmed,
        "independent_support": independent_support,
        "renderer_admitted": renderer_admitted,
        "geometry_persistent": geometry_persistent,
        "initial_opacity": initial_opacity,
        "scene_envelope": scene_envelope,
        "sampling_footprint_multiplier": sampling_multiplier,
    }


def build_surface_seed(
    evidence_store: Path,
    output: Path,
    *,
    rgb_root: Path | None = None,
    minimum_rigid_probability: float = 0.70,
    maximum_canopy_probability: float = 0.15,
    consensus_radius: float = 0.10,
    dedup_radius: float = 0.012,
    chart_seeds_per_view: int = 3000,
    maximum_chart_seeds: int = 120_000,
    mast3r_pointmap_seeds_per_view: int = 6000,
    maximum_mast3r_pointmap_seeds: int = 240_000,
    mast3r_pointmap_minimum_confidence: float = 1.25,
    mast3r_cross_sequence_radius: float = 0.15,
    mast3r_pointmap_voxel_size: float = 0.018,
    mast3r_maximum_single_sequence_fraction: float = 0.25,
    maximum_dav2_rigid_seeds: int = 0,
    dav2_rigid_selected_views: int = 0,
    dav2_rigid_seeds_per_view: int = 256,
    dav2_rigid_cross_sequence_radius: float = 0.25,
    seed: int = 991,
) -> dict[str, Any]:
    store = load_evidence_store(evidence_store)
    dataset = Path(store["dataset"]).expanduser().resolve()
    rgb_root = (
        Path(rgb_root).expanduser().resolve()
        if rgb_root is not None
        else (dataset / "images").resolve()
    )
    if not rgb_root.is_dir():
        raise FileNotFoundError(
            f"Surface target RGB root does not exist: {rgb_root}"
        )
    mast3r_only = store.get("geometry_source") == "mast3r_only"
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    track_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    chart_id_parts: list[np.ndarray] = []
    chart_uv_parts: list[np.ndarray] = []
    pointmap_view_parts: list[np.ndarray] = []
    pointmap_cross_sequence_parts: list[np.ndarray] = []
    persistent_geometry_parts: list[np.ndarray] = []
    chart_independent_support_parts: list[np.ndarray] = []
    rigid_metric_reference_parts: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    dense_pointmap = None
    dav2_rigid = None

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
        pointmap_view_parts.append(
            np.full(int(colmap_keep.sum()), -1, dtype=np.int32)
        )
        pointmap_cross_sequence_parts.append(
            np.zeros(int(colmap_keep.sum()), dtype=bool)
        )
        persistent_geometry_parts.append(
            np.ones(int(colmap_keep.sum()), dtype=bool)
        )
        chart_independent_support_parts.append(
            np.zeros(int(colmap_keep.sum()), dtype=bool)
        )
        rigid_metric_reference_parts.append(colmap["xyz"][colmap_keep])
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
        if mast3r_only:
            # A same-sequence source-star is not a stable building landmark.
            # Surface initialization uses only tracks independently observed
            # across Cambridge traversals; sequence-local canopy evidence is
            # retained for the volume branch below.
            mast3r_keep &= mast3r["sequence_count"] >= 2
            if "cycle_consistency" in mast3r:
                mast3r_keep &= mast3r["cycle_consistency"] >= 0.50
            if "pointmap_spread" in mast3r:
                mast3r_keep &= mast3r["pointmap_spread"] <= 0.08
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
        pointmap_view_parts.append(
            np.full(len(mast3r_indices), -1, dtype=np.int32)
        )
        pointmap_cross_sequence_parts.append(
            np.zeros(len(mast3r_indices), dtype=bool)
        )
        persistent_geometry_parts.append(
            np.ones(len(mast3r_indices), dtype=bool)
        )
        chart_independent_support_parts.append(
            np.zeros(len(mast3r_indices), dtype=bool)
        )
        rigid_metric_reference_parts.append(
            mast3r["xyz"][mast3r_indices]
        )
        source_counts["mast3r_rigid"] = int(len(mast3r_indices))

    pointmap_index = artifact_path(
        store, "mast3r_pointmap_index", required=False
    )
    if pointmap_index is not None:
        semantic_payload = json.loads(
            Path(store["semantic_contract"]).read_text(encoding="utf-8")
        )
        dense_pointmap = _sample_dense_mast3r_rigid_seeds(
            store,
            dataset=dataset,
            tree_mask_pickle=Path(semantic_payload["tree_mask_pickle"]),
            samples_per_view=mast3r_pointmap_seeds_per_view,
            maximum_total=maximum_mast3r_pointmap_seeds,
            minimum_confidence=mast3r_pointmap_minimum_confidence,
            cross_sequence_radius=mast3r_cross_sequence_radius,
            voxel_size=mast3r_pointmap_voxel_size,
            maximum_single_sequence_fraction=(
                mast3r_maximum_single_sequence_fraction
            ),
        )
        reference = (
            np.concatenate(xyz_parts)
            if xyz_parts and sum(map(len, xyz_parts))
            else np.empty((0, 3), dtype=np.float32)
        )
        novel = _deduplicate(
            dense_pointmap["xyz"], reference, radius=dedup_radius
        )
        for key in dense_pointmap:
            dense_pointmap[key] = dense_pointmap[key][novel]
        count = len(dense_pointmap["xyz"])
        xyz_parts.append(dense_pointmap["xyz"])
        rgb_parts.append(dense_pointmap["rgb"])
        sigma_parts.append(dense_pointmap["sigma"])
        confidence_parts.append(dense_pointmap["confidence"])
        track_parts.append(np.full(count, -1, dtype=np.int64))
        source_parts.append(
            np.full(count, SOURCE_MAST3R, dtype=np.int8)
        )
        chart_id_parts.append(np.full(count, -1, dtype=np.int32))
        chart_uv_parts.append(dense_pointmap["uv"])
        pointmap_view_parts.append(dense_pointmap["view_id"])
        pointmap_cross_sequence_parts.append(
            dense_pointmap["cross_sequence_supported"]
        )
        # Pointmaps remain immutable observation factors, not permanently
        # protected Gaussian centres.
        persistent_geometry_parts.append(np.zeros(count, dtype=bool))
        chart_independent_support_parts.append(
            np.zeros(count, dtype=bool)
        )
        rigid_metric_reference_parts.append(
            dense_pointmap["xyz"][
                dense_pointmap["cross_sequence_supported"]
            ]
        )
        source_counts["mast3r_dense_rigid"] = int(count)
        source_counts["mast3r_dense_cross_sequence"] = int(
            dense_pointmap["cross_sequence_supported"].sum()
        )
        source_counts["mast3r_dense_single_sequence"] = int(
            (~dense_pointmap["cross_sequence_supported"]).sum()
        )
    if mast3r_only and not sum(map(len, xyz_parts)):
        raise RuntimeError(
            "MASt3R-only initialization has no valid rigid observations"
        )

    chart_path = artifact_path(
        store, "chart_geometry", required=False
    )
    chart_cameras = artifact_path(
        store, "chart_cameras", required=False
    )
    chart_consensus = artifact_path(
        store, "chart_crossview_consensus", required=False
    )
    if chart_path is not None and chart_cameras is not None:
        if mast3r_only and chart_consensus is None:
            raise RuntimeError(
                "MASt3R-only initialization requires pixelwise Chart "
                "cross-view consensus"
            )
        chart = _sample_chart_seeds(
            chart_path,
            chart_cameras,
            chart_consensus=chart_consensus,
            scene_contract=Path(store["scene_contract"]),
            dataset=Path(store["dataset"]),
            tree_mask_pickle=Path(
                json.loads(
                    Path(store["semantic_contract"]).read_text(
                        encoding="utf-8"
                    )
                )["tree_mask_pickle"]
            ),
            rigid_reference=(
                np.concatenate(rigid_metric_reference_parts)
                if rigid_metric_reference_parts
                and sum(map(len, rigid_metric_reference_parts))
                else np.empty((0, 3), dtype=np.float32)
            ),
            per_chart=chart_seeds_per_view,
            maximum_total=maximum_chart_seeds,
            consensus_radius=consensus_radius,
            seed=seed,
        )
        chart_candidate_count = len(chart["xyz"])
        chart_candidate_supported = int(
            chart["independent_support"].sum()
        )
        chart_candidate_confirmed = int(
            chart["consensus_confirmed"].sum()
        )
        novel = _deduplicate(
            chart["xyz"],
            np.concatenate(xyz_parts),
            radius=dedup_radius,
        )
        # A Chart raster is consumed in full by the native ray/depth factor.
        # Primitive admission is a separate ownership decision.  The v38/v39
        # path computed this mask but never consumed it, accidentally adding
        # 128k unsupported sheets to the renderer.
        novel &= chart["renderer_admitted"].astype(bool)
        chart_row_count = len(chart["xyz"])
        for key, value in tuple(chart.items()):
            if (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == chart_row_count
            ):
                chart[key] = value[novel]
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
        pointmap_view_parts.append(
            np.full(len(chart["xyz"]), -1, dtype=np.int32)
        )
        pointmap_cross_sequence_parts.append(
            np.zeros(len(chart["xyz"]), dtype=bool)
        )
        # Unsupported Chart samples are coverage-only renderer witnesses.
        # They may receive RGB gradients during bootstrap, but only pixels
        # confirmed by independent views may act as geometry observations.
        persistent_geometry_parts.append(
            chart["geometry_persistent"].astype(bool)
        )
        chart_independent_support_parts.append(
            chart["independent_support"].astype(bool)
        )
        source_counts["chart_rigid_residual"] = int(len(chart["xyz"]))
        source_counts["chart_renderer_candidate_total"] = int(
            chart_candidate_count
        )
        source_counts["chart_renderer_candidates_supported"] = int(
            chart_candidate_supported
        )
        source_counts["chart_renderer_candidates_confirmed"] = int(
            chart_candidate_confirmed
        )
        source_counts["chart_renderer_candidates_not_admitted"] = int(
            chart_candidate_count - chart_candidate_supported
        )
        source_counts["chart_crossview_confirmed"] = int(
            chart["consensus_confirmed"].sum()
        )
        source_counts["chart_independently_supported"] = int(
            chart["independent_support"].sum()
        )
        source_counts["chart_low_opacity_coverage"] = int(
            (~chart["independent_support"]).sum()
        )
        source_counts["chart_gross_depth_outliers_rejected"] = int(
            chart["scene_envelope"]["rejected"]
        )
        source_counts["chart_unsupported_bootstrap"] = int(
            (~chart["consensus_confirmed"]).sum()
        )

    if (
        int(maximum_dav2_rigid_seeds) > 0
        and int(dav2_rigid_selected_views) > 0
    ):
        semantic_payload = json.loads(
            Path(store["semantic_contract"]).read_text(encoding="utf-8")
        )
        rigid_reference = np.concatenate(xyz_parts).astype(np.float32)
        dav2_rigid = _dav2_rigid_hole_completion_samples(
            store,
            dataset=dataset,
            rgb_root=rgb_root,
            tree_mask_pickle=Path(
                semantic_payload["tree_mask_pickle"]
            ),
            rigid_xyz=rigid_reference,
            selected_view_count=dav2_rigid_selected_views,
            samples_per_view=dav2_rigid_seeds_per_view,
            maximum_total=maximum_dav2_rigid_seeds,
            cross_sequence_radius=dav2_rigid_cross_sequence_radius,
        )
        count = len(dav2_rigid["xyz"])
        if count:
            xyz_parts.append(dav2_rigid["xyz"])
            rgb_parts.append(dav2_rigid["rgb"])
            sigma_parts.append(dav2_rigid["sigma"])
            confidence_parts.append(dav2_rigid["confidence"])
            track_parts.append(
                np.full(count, -1, dtype=np.int64)
            )
            source_parts.append(
                np.full(count, SOURCE_SURFACE_DAV2, dtype=np.int8)
            )
            chart_id_parts.append(
                np.full(count, -1, dtype=np.int32)
            )
            chart_uv_parts.append(
                np.full((count, 2), np.nan, dtype=np.float32)
            )
            pointmap_view_parts.append(
                np.full(count, -1, dtype=np.int32)
            )
            pointmap_cross_sequence_parts.append(
                np.zeros(count, dtype=bool)
            )
            # These are renderer births calibrated by the permanent
            # MASt3R/MAtCha scaffold.  DAV2 remains an ordinal factor and
            # never becomes a permanent metric observation by itself.
            persistent_geometry_parts.append(
                np.zeros(count, dtype=bool)
            )
            chart_independent_support_parts.append(
                np.zeros(count, dtype=bool)
            )
        source_counts["dav2_rigid_hole_completion"] = int(count)

    xyz = np.concatenate(xyz_parts).astype(np.float32)
    rgb = np.concatenate(rgb_parts).astype(np.float32)
    sigma = np.concatenate(sigma_parts).astype(np.float32)
    confidence = np.concatenate(confidence_parts).astype(np.float32)
    track_id = np.concatenate(track_parts).astype(np.int64)
    source_type = np.concatenate(source_parts).astype(np.int8)
    chart_id = np.concatenate(chart_id_parts).astype(np.int32)
    chart_uv = np.concatenate(chart_uv_parts).astype(np.float32)
    pointmap_view_id = np.concatenate(pointmap_view_parts).astype(np.int32)
    pointmap_cross_sequence_supported = np.concatenate(
        pointmap_cross_sequence_parts
    ).astype(bool)
    persistent_geometry_evidence = np.concatenate(
        persistent_geometry_parts
    ).astype(bool)
    chart_independent_support = np.concatenate(
        chart_independent_support_parts
    ).astype(bool)
    scales = np.empty((len(xyz), 2), dtype=np.float32)
    quaternions = np.empty((len(xyz), 4), dtype=np.float32)
    normals = np.empty((len(xyz), 3), dtype=np.float32)
    free = source_type != SOURCE_CHART
    if bool(free.any()):
        scales[free], quaternions[free], normals[free] = _surface_frames(
            xyz[free], sigma[free], maximum_scale=0.08
        )
    dense_rows = pointmap_view_id >= 0
    if bool(dense_rows.any()):
        if dense_pointmap is None or int(dense_rows.sum()) != len(
            dense_pointmap["xyz"]
        ):
            raise RuntimeError(
                "Dense MASt3R frame metadata no longer matches surface rows"
            )
        scales[dense_rows] = dense_pointmap["scales"]
        quaternions[dense_rows] = dense_pointmap["quaternions"]
        normals[dense_rows] = dense_pointmap["normals"]
    chart_rows = source_type == SOURCE_CHART
    if bool(chart_rows.any()):
        scales[chart_rows] = chart["scales"]
        quaternions[chart_rows] = chart["quaternions"]
        normals[chart_rows] = chart["normals"]
    dav2_rows = source_type == SOURCE_SURFACE_DAV2
    if bool(dav2_rows.any()):
        if dav2_rigid is None or int(dav2_rows.sum()) != len(
            dav2_rigid["xyz"]
        ):
            raise RuntimeError(
                "DAV2 rigid completion metadata no longer matches surface "
                "rows"
            )
        scales[dav2_rows] = dav2_rigid["scales"]
        quaternions[dav2_rows] = dav2_rigid["quaternions"]
        normals[dav2_rows] = dav2_rigid["normals"]
    # Opacity encodes initialization authority, not merely source identity.
    # Multi-view tracks and high-precision pointmap pixels are metric seeds;
    # unsupported Chart pixels remain visible as coverage witnesses but must
    # not form an opaque displaced sheet before RGB/geometry can adjudicate
    # them.
    initial_opacity = np.full(len(xyz), 0.025, dtype=np.float32)
    initial_opacity[source_type == SOURCE_COLMAP] = 0.08
    measured_tracks = (source_type == SOURCE_MAST3R) & (track_id >= 0)
    initial_opacity[measured_tracks] = 0.08
    if bool(dense_rows.any()):
        dense_precision = dense_pointmap[
            "cross_sequence_precision"
        ].astype(np.float32)
        dense_supported = dense_pointmap[
            "cross_sequence_supported"
        ].astype(bool)
        dense_opacity = np.full(len(dense_precision), 0.025, dtype=np.float32)
        dense_opacity[dense_supported] = (
            0.045 + 0.020 * dense_precision[dense_supported]
        )
        initial_opacity[dense_rows] = dense_opacity
    if bool(chart_rows.any()):
        initial_opacity[chart_rows] = chart["initial_opacity"]
    if bool(dav2_rows.any()):
        initial_opacity[dav2_rows] = dav2_rigid["initial_opacity"]
    dav2_observation_view_name = np.full(
        len(xyz), "", dtype="<U128"
    )
    dav2_observation_uv = np.full(
        (len(xyz), 2), np.nan, dtype=np.float32
    )
    dav2_observation_depth = np.full(
        len(xyz), np.nan, dtype=np.float32
    )
    if bool(dav2_rows.any()):
        dav2_observation_view_name[dav2_rows] = dav2_rigid[
            "observation_view_name"
        ]
        dav2_observation_uv[dav2_rows] = dav2_rigid[
            "observation_uv"
        ]
        dav2_observation_depth[dav2_rows] = dav2_rigid[
            "observation_depth"
        ]
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
        pointmap_view_id=pointmap_view_id,
        pointmap_cross_sequence_supported=(
            pointmap_cross_sequence_supported
        ),
        persistent_geometry_evidence=persistent_geometry_evidence,
        chart_independent_support=chart_independent_support,
        dav2_observation_view_name=dav2_observation_view_name,
        dav2_observation_uv=dav2_observation_uv,
        dav2_observation_depth=dav2_observation_depth,
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
        "persistent_geometry_seed_count": int(
            persistent_geometry_evidence.sum()
        ),
        "coverage_only_bootstrap_seed_count": int(
            (~persistent_geometry_evidence).sum()
        ),
        "mast3r_pointmap_contract": {
            "samples_per_view": int(mast3r_pointmap_seeds_per_view),
            "maximum_total": int(maximum_mast3r_pointmap_seeds),
            "minimum_confidence": float(
                mast3r_pointmap_minimum_confidence
            ),
            "cross_sequence_radius": float(
                mast3r_cross_sequence_radius
            ),
            "voxel_size": float(mast3r_pointmap_voxel_size),
            "single_sequence_is_uncertainty_not_deletion": True,
            "maximum_single_sequence_seed_fraction": float(
                mast3r_maximum_single_sequence_fraction
            ),
            "footprint_policy": (
                "native_tangent_times_retained_pixel_nearest_spacing_over_two"
            ),
            "footprint_multiplier_percentiles": (
                np.percentile(
                    dense_pointmap["sampling_footprint_multiplier"],
                    [0, 10, 50, 90, 100],
                ).tolist()
                if dense_pointmap is not None
                and len(dense_pointmap["sampling_footprint_multiplier"])
                else []
            ),
            "posterior_precision_percentiles": (
                np.percentile(
                    dense_pointmap["cross_sequence_precision"],
                    [0, 10, 50, 90, 100],
                ).tolist()
                if dense_pointmap is not None
                and len(dense_pointmap["cross_sequence_precision"])
                else []
            ),
            "initialization_consumes_native_posterior": True,
            "camera_ray_policy": (
                "preserve_camera_z_then_unproject_exact_fixed_K"
            ),
        },
        "chart_sampling_contract": {
            "footprint_policy": (
                "native_tangent_times_trust_weighted_retained_uv_spacing"
            ),
            "coverage_admission": (
                "full_chart_raster_remains_external_factor__renderer_birth_"
                "requires_independent_metric_or_cross_chart_support"
            ),
            "unsupported_chart_is_renderer_primitive": False,
            "unsupported_chart_remains_native_observation": True,
            "gross_depth_scene_envelope": chart["scene_envelope"],
            "camera_ray_policy": (
                "preserve_camera_z_then_unproject_exact_fixed_K"
            ),
            "footprint_multiplier_percentiles": (
                np.percentile(
                    chart["sampling_footprint_multiplier"],
                    [0, 10, 50, 90, 100],
                ).tolist()
                if chart_path is not None
                and chart_cameras is not None
                and len(chart["sampling_footprint_multiplier"])
                else []
            ),
        },
        "dav2_rigid_hole_completion": (
            dav2_rigid["audit"]
            if dav2_rigid is not None
            else {
                "enabled": False,
                "reason": "maximum_seed_or_view_budget_is_zero",
                "selected": 0,
            }
        ),
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


def _filter_sequence_local_scene_envelope(
    samples: list[dict[str, Any]],
    rigid_xyz: np.ndarray,
    *,
    expansion: float = 1.35,
    minimum_margin: float = 5.0,
) -> tuple[list[dict[str, Any]], int, dict[str, float]]:
    """Remove only gross one-view depth outliers from foliage proposals.

    The envelope uses a 99.9th-percentile rigid-scene radius and is expanded
    substantially, so it is not a tight building mask. Its purpose is to stop
    one-frame DAV2/pointmap failures hundreds of metres away from chaining
    otherwise local tree instances together. Native multi-view tracks do not
    pass through this filter.
    """
    if not samples:
        return [], 0, {
            "reference_radius": 0.0,
            "maximum_radius": 0.0,
        }
    rigid_xyz = np.asarray(rigid_xyz, dtype=np.float64)
    center = np.median(rigid_xyz, axis=0)
    rigid_radius = np.linalg.norm(rigid_xyz - center[None], axis=1)
    reference_radius = float(np.quantile(rigid_radius, 0.999))
    maximum_radius = max(
        reference_radius * float(expansion),
        reference_radius + float(minimum_margin),
    )
    sample_xyz = np.stack([sample["xyz"] for sample in samples])
    keep = (
        np.isfinite(sample_xyz).all(axis=1)
        & (
            np.linalg.norm(sample_xyz - center[None], axis=1)
            <= maximum_radius
        )
    )
    retained = [
        sample for sample, accepted in zip(samples, keep) if accepted
    ]
    return retained, int((~keep).sum()), {
        "reference_radius": reference_radius,
        "maximum_radius": maximum_radius,
    }


def _align_tracks_to_local_hull_instances(
    hull_centers: np.ndarray,
    hull_instance_ids: np.ndarray,
    track_xyz: np.ndarray,
    *,
    association_radius: float = 1.5,
    maximum_component_extent: float = 12.0,
    persistent_track_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use accepted local hull components as tree-ownership authorities.

    Mapping every hull cell to the nearest one of a few sparse native-track
    components recreated 100+ metre "tree instances" after an otherwise
    bounded clustering pass. Preserve the independently accepted hull
    components, associate only nearby sequence-local observations, and give
    the remaining dynamic-only observations their own bounded components.
    """
    hull_centers = np.asarray(hull_centers, dtype=np.float64).reshape(-1, 3)
    hull_instance_ids = np.asarray(
        hull_instance_ids, dtype=np.int32
    ).reshape(-1)
    track_xyz = np.asarray(track_xyz, dtype=np.float64).reshape(-1, 3)
    if len(hull_centers) != len(hull_instance_ids):
        raise ValueError("Hull centre/instance arrays have different lengths")
    if persistent_track_count is None:
        persistent_track_count = len(track_xyz)
    persistent_track_count = int(persistent_track_count)
    if not 0 <= persistent_track_count <= len(track_xyz):
        raise ValueError(
            "persistent_track_count must lie inside the track table"
        )

    # Candidate dilation can leave disconnected accepted islands under one
    # upstream anchor label. Refine labels without deleting any geometry.
    local_hull_ids = np.full(len(hull_centers), -1, dtype=np.int32)
    next_id = 0
    for source_id in np.unique(hull_instance_ids):
        rows = np.flatnonzero(hull_instance_ids == source_id)
        labels = cluster_tree_instances(
            hull_centers[rows],
            maximum_component_extent=maximum_component_extent,
        )
        local_hull_ids[rows] = labels + next_id
        next_id += int(len(np.unique(labels)))
    hull_local_instance_count = int(next_id)

    if len(hull_centers):
        distance, nearest = cKDTree(hull_centers).query(track_xyz, k=1)
        associated = distance <= float(association_radius)
    else:
        distance = np.full(len(track_xyz), np.inf, dtype=np.float64)
        nearest = np.zeros(len(track_xyz), dtype=np.int64)
        associated = np.zeros(len(track_xyz), dtype=bool)
    track_instance_ids = np.full(len(track_xyz), -1, dtype=np.int32)
    if bool(associated.any()):
        track_instance_ids[associated] = local_hull_ids[
            nearest[associated]
        ]

    # Renderer-basis rows are incremental consumers of an already measured
    # ownership graph.  They may inherit a nearby persistent instance or
    # form new bounded components, but changing their sampling density must
    # never relabel the persistent MASt3R/Chart/DAV2 rows that define the
    # canonical and trunk contracts.
    row_index = np.arange(len(track_xyz))
    persistent_dynamic_only = np.flatnonzero(
        (~associated) & (row_index < persistent_track_count)
    )
    if len(persistent_dynamic_only):
        labels = cluster_tree_instances(
            track_xyz[persistent_dynamic_only],
            maximum_component_extent=maximum_component_extent,
        )
        track_instance_ids[persistent_dynamic_only] = labels + next_id
        next_id += int(len(np.unique(labels)))

    incremental = np.flatnonzero(row_index >= persistent_track_count)
    incremental_unassociated = incremental[~associated[incremental]]
    inherited_incremental = np.empty(0, dtype=np.int64)
    if len(incremental_unassociated) and persistent_track_count:
        inherited_distance, inherited_nearest = cKDTree(
            track_xyz[:persistent_track_count]
        ).query(track_xyz[incremental_unassociated], k=1)
        inherit = inherited_distance <= float(association_radius)
        inherited_incremental = incremental_unassociated[inherit]
        track_instance_ids[inherited_incremental] = track_instance_ids[
            inherited_nearest[inherit]
        ]
        incremental_unassociated = incremental_unassociated[~inherit]
    incremental_new_instance_count = 0
    if len(incremental_unassociated):
        labels = cluster_tree_instances(
            track_xyz[incremental_unassociated],
            maximum_component_extent=maximum_component_extent,
        )
        track_instance_ids[incremental_unassociated] = labels + next_id
        incremental_new_instance_count = int(len(np.unique(labels)))
        next_id += incremental_new_instance_count
    if bool((track_instance_ids < 0).any()):
        raise RuntimeError("Track-instance extension left unassigned rows")

    dynamic_only = np.flatnonzero(~associated)
    dynamic_only_instance_count = int(
        len(np.unique(track_instance_ids[dynamic_only]))
    ) if len(dynamic_only) else 0
    return local_hull_ids, track_instance_ids, {
        "association_radius": float(association_radius),
        "maximum_component_extent": float(maximum_component_extent),
        "hull_local_instance_count": hull_local_instance_count,
        "associated_track_count": int(associated.sum()),
        "dynamic_only_track_count": int((~associated).sum()),
        "dynamic_only_instance_count": dynamic_only_instance_count,
        "persistent_track_count": persistent_track_count,
        "incremental_track_count": int(
            len(track_xyz) - persistent_track_count
        ),
        "incremental_inherited_track_count": int(
            len(inherited_incremental)
        ),
        "incremental_new_instance_count": int(
            incremental_new_instance_count
        ),
        "incremental_ownership_contract": (
            "inherit_nearby_persistent_instance_else_new_bounded_component_"
            "never_relabel_persistent_rows"
        ),
        "association_distance_median": (
            float(np.median(distance)) if len(hull_centers) else float("nan")
        ),
        "association_distance_p99": (
            float(np.quantile(distance, 0.99))
            if len(hull_centers)
            else float("nan")
        ),
    }


def _nonchart_pointmap_foliage_samples(
    store: dict,
    images: dict[int, dict],
    masks: CambridgeMaskLookup,
    *,
    exclude_stems: set[str] | None = None,
    samples_per_view: int = 800,
    maximum_samples: int = 48_000,
) -> list[dict[str, Any]]:
    """Retain canopy observations from pointmaps outside the Chart subset.

    MAtCha intentionally operates on a compact set of surface Charts, whereas
    complementary MASt3R runs can cover many additional fixed-camera views.
    The former implementation indexed all those pointmaps for rigid seeds but
    silently iterated only ``chart_cameras.filepaths`` when constructing
    foliage.  Consequently an added tree view could pass every Evidence Store
    hash check and still have zero effect on the canopy model.

    These rows remain sequence-local observations.  They may propose
    supplemental visual-hull cells and ray/depth posteriors, but canonical
    acceptance still requires agreement from independent traversals.
    """
    index_path = artifact_path(
        store, "mast3r_pointmap_index", required=False
    )
    if index_path is None:
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("coordinate_frame") != "cambridge_fixed_world":
        raise RuntimeError(
            "Canopy pointmaps are not in the fixed Cambridge world"
        )
    records = payload.get("records", {})
    camera_order = payload.get("camera_order", list(records))
    if set(camera_order) != set(records):
        raise RuntimeError("MASt3R pointmap camera order is incomplete")
    excluded = {Path(value).stem for value in (exclude_stems or set())}
    by_stem = {
        Path(str(image["name"])).stem: int(image_id)
        for image_id, image in images.items()
    }
    scene_payload = json.loads(
        Path(store["scene_contract"]).read_text(encoding="utf-8")
    )
    scene_records = {
        Path(str(row["image_name"])).stem: row
        for row in scene_payload["records"]
    }
    dataset = Path(store["dataset"])
    result: list[dict[str, Any]] = []
    for stem in camera_order:
        if stem in excluded:
            continue
        image_id = by_stem.get(stem)
        scene_record = scene_records.get(stem)
        if image_id is None or scene_record is None:
            raise RuntimeError(
                f"Pointmap canopy view is absent from the scene: {stem}"
            )
        record = records[stem]
        pointmap_path = Path(record["path"])
        if not pointmap_path.is_file():
            raise FileNotFoundError(pointmap_path)
        point_flat, confidence_flat = _load_mast3r_pointmap_geometry(
            pointmap_path
        )
        image_path = dataset / "images" / images[image_id]["name"]
        with Image.open(image_path) as handle:
            source_size = handle.size
        height, width = _native_pointmap_shape(
            len(confidence_flat), source_size
        )
        points = _retarget_pointmap_fixed_camera_rays(
            point_flat,
            (height, width),
            scene_record,
        ).reshape(height, width, 3)
        confidence = confidence_flat.reshape(height, width)
        key = masks.source_name_for(images[image_id]["name"])
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
            & np.isfinite(confidence)
            & (confidence >= 1.0)
            & np.isfinite(points).all(axis=-1)
            & (np.linalg.norm(points, axis=-1) > 1e-5)
        )
        candidates = np.flatnonzero(canopy)
        if not len(candidates):
            continue
        posterior_precision = np.full(
            (height, width), 0.03, dtype=np.float32
        )
        posterior_supported = np.zeros(
            (height, width), dtype=bool
        )
        posterior_record = record.get("cross_sequence_posterior")
        if posterior_record is not None:
            posterior_path = Path(posterior_record["path"])
            with np.load(posterior_path, allow_pickle=False) as archive:
                native_shape = tuple(
                    map(int, archive["native_shape"])
                )
                if native_shape != (height, width):
                    raise RuntimeError(
                        "Pointmap foliage posterior shape mismatch for "
                        f"{stem}: {native_shape} != {(height, width)}"
                    )
                posterior_precision = archive[
                    "precision"
                ].astype(np.float32)
                posterior_supported = archive[
                    "cross_sequence_supported"
                ].astype(bool)
        confidence_score = (
            confidence
            * np.sqrt(np.clip(posterior_precision, 0.03, 1.0))
        ).reshape(-1)
        ranked = candidates[
            np.argsort(
                confidence_score[candidates], kind="stable"
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
        coverage = ranked[np.sort(first)]
        if len(coverage) < int(samples_per_view):
            detail_mask = np.ones(len(ranked), dtype=bool)
            detail_mask[np.sort(first)] = False
            coverage = np.concatenate(
                [
                    coverage,
                    ranked[detail_mask][
                        : int(samples_per_view) - len(coverage)
                    ],
                ]
            )
        selected = coverage[: int(samples_per_view)]
        with Image.open(image_path) as handle:
            rgb = np.asarray(
                handle.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            ).reshape(-1, 3)
        rotation = quaternion_to_rotation(images[image_id]["qvec"])
        translation = np.asarray(
            images[image_id]["tvec"], dtype=np.float64
        )
        selected_xyz = points.reshape(-1, 3)[selected].astype(
            np.float64
        )
        selected_depth = (
            selected_xyz @ rotation.T + translation[None]
        )[:, 2]
        selected_precision = posterior_precision.reshape(-1)[selected]
        selected_supported = posterior_supported.reshape(-1)[selected]
        selected_confidence = confidence.reshape(-1)[selected]
        for offset, flat in enumerate(selected):
            result.append(
                {
                    "id": -(20_000_000 + len(result)),
                    "xyz": selected_xyz[offset],
                    "rgb": rgb[flat],
                    "error": float(
                        np.clip(
                            0.5
                            / max(
                                np.sqrt(
                                    float(selected_confidence[offset])
                                    * float(selected_precision[offset])
                                ),
                                0.25,
                            ),
                            0.15,
                            3.0,
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
                        [selected_depth[offset]], dtype=np.float32
                    ),
                    "tree_sequence_count": 1,
                    "tree_fraction": float(
                        0.65
                        + 0.35
                        * np.clip(
                            selected_precision[offset], 0.0, 1.0
                        )
                    ),
                    "pointmap_sequence_local_sample": True,
                    "pointmap_cross_sequence_supported": bool(
                        selected_supported[offset]
                    ),
                }
            )
    # Apply the global cap only after every indexed fixed camera has had a
    # chance to contribute. The former early break made camera-order prefixes
    # define the sequence-local canopy evidence.
    result, _ = _balanced_single_camera_sample_cap(
        result,
        maximum_samples,
        source_name="MASt3R pointmap",
    )
    return result


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
    scene_payload = json.loads(
        Path(store["scene_contract"]).read_text(encoding="utf-8")
    )
    scene_records = {
        Path(str(row["image_name"])).stem: row
        for row in scene_payload["records"]
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
            scene_record = scene_records.get(image_path.stem)
            if scene_record is None:
                raise RuntimeError(
                    "Chart pointmap is missing from the fixed scene "
                    f"contract: {image_path.stem}"
                )
            raw_xyz = _retarget_pointmap_fixed_camera_rays(
                raw_xyz,
                (height, width),
                scene_record,
            ).reshape(height, width, 3)
            aligned_xyz = (
                chart_points[int(chart_index)].astype(np.float32)
                / scale_factor
            )
            aligned_xyz = _retarget_pointmap_fixed_camera_rays(
                aligned_xyz,
                (height, width),
                scene_record,
            ).reshape(height, width, 3)
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
    allowed_image_ids: set[int] | None = None,
    rigid_depth_maps: dict[int, np.ndarray] | None = None,
    samples_per_view: int = 128,
    maximum_samples: int = 120_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    weak_posterior_views = 0
    rejected_views = 0
    posterior_confidences: list[float] = []
    posterior_relative_sigmas: list[float] = []
    rigid_xyz = np.asarray(rigid_xyz, dtype=np.float64)
    rigid_depth_maps = rigid_depth_maps or {}
    dataset = Path(store["dataset"])

    eligible_image_ids = (
        {int(value) for value in allowed_image_ids}
        if allowed_image_ids is not None
        else {int(value) for value in images}
    )
    visited_views = 0
    for image_id in sorted(images):
        if int(image_id) not in eligible_image_ids:
            continue
        visited_views += 1
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
        rendered_rigid_depth = rigid_depth_maps.get(int(image_id))
        if rendered_rigid_depth is not None:
            zbuffer = np.asarray(
                rendered_rigid_depth, dtype=np.float32
            )
            if zbuffer.shape != (height, width):
                finite = np.isfinite(zbuffer) & (zbuffer > 0)
                values = np.where(finite, zbuffer, 0.0)
                values = torch.nn.functional.interpolate(
                    torch.from_numpy(values)[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0].numpy()
                finite = torch.nn.functional.interpolate(
                    torch.from_numpy(finite.astype(np.float32))[None, None],
                    size=(height, width),
                    mode="nearest",
                )[0, 0].numpy() > 0.5
                zbuffer = np.where(finite, values, np.inf).astype(
                    np.float32
                )
            calibration_depth_source = "trained_rigid_surface_render"
        else:
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
            calibration_depth_source = "sparse_surface_seed_zbuffer"
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
        posterior_status = "accepted"
        relative_sigma = float(diagnostics.relative_rmse)
        if not diagnostics.accepted:
            # A finite positive affine map remains useful as a spatially
            # uncertain ray posterior even when its residual exceeds the
            # old binary quality threshold.  This is especially important in
            # tree-dominated frames, where only a narrow rigid strip is
            # visible.  Invalid scale/sign and insufficient support remain
            # physically undefined and are still rejected.
            if (
                diagnostics.support_pixels >= 64
                and np.isfinite(diagnostics.alpha)
                and np.isfinite(diagnostics.beta)
                and diagnostics.beta > 0
                and np.isfinite(diagnostics.relative_rmse)
            ):
                denominator = (
                    float(diagnostics.alpha)
                    + float(diagnostics.beta)
                    / np.maximum(relative_depth, 1e-6)
                )
                positive = (
                    np.isfinite(relative_depth)
                    & (relative_depth > 0)
                    & np.isfinite(denominator)
                    & (denominator > 1e-6)
                )
                aligned = torch.zeros_like(
                    torch.from_numpy(relative_depth)
                )
                aligned.numpy()[positive] = (
                    1.0 / denominator[positive]
                ).astype(np.float32)
                supported_depths = zbuffer[support]
                lower = max(
                    float(np.quantile(supported_depths, 0.01)) / 4.0,
                    1e-4,
                )
                upper = max(
                    float(np.quantile(supported_depths, 0.99)) * 4.0,
                    lower * 2.0,
                )
                aligned[torch.from_numpy(positive)] = aligned[
                    torch.from_numpy(positive)
                ].clamp(min=lower, max=upper)
                posterior_status = "weak_continuous"
            else:
                rejected_views += 1
                continue
        support_confidence = min(
            diagnostics.support_pixels / 512.0, 1.0
        )
        posterior_confidence = float(
            np.clip(
                support_confidence
                * float(diagnostics.inlier_ratio)
                * np.exp(
                    -0.5
                    * (
                        max(float(diagnostics.relative_rmse), 0.0)
                        / 0.35
                    )
                    ** 2
                ),
                0.01,
                1.0,
            )
        )
        relative_sigma = float(
            np.clip(max(relative_sigma, 0.05), 0.05, 1.50)
        )
        if not np.isfinite(posterior_confidence):
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
        if posterior_status == "weak_continuous":
            weak_posterior_views += 1
        posterior_confidences.append(posterior_confidence)
        posterior_relative_sigmas.append(relative_sigma)
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
            absolute_depth_sigma = float(
                np.clip(
                    relative_sigma * float(depth_value),
                    0.05,
                    3.0,
                )
            )
            output.append(
                {
                    "id": -(10_000_000 + len(output)),
                    "xyz": point.astype(np.float64),
                    "rgb": rgb_image[int(row), int(column)],
                    "error": float(
                        max(diagnostics.relative_rmse, 0.02)
                    ),
                    "position_sigma": absolute_depth_sigma,
                    "observation_depth_sigma": (
                        absolute_depth_sigma
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
                        posterior_confidence
                    ),
                    "ray_footprint_scale": float(point_scale),
                    "dav2_sequence_local_sample": True,
                    "dav2_alignment_posterior": (
                        posterior_confidence
                    ),
                    "ray_depth_nll": float(
                        1.0
                        / max(posterior_confidence, 1.0e-4) ** 2
                        - 1.0
                    ),
                    "dav2_alignment_status": posterior_status,
                    "dav2_calibration_depth_source": (
                        calibration_depth_source
                    ),
                }
            )
    output, sample_budget_audit = _balanced_single_camera_sample_cap(
        output,
        maximum_samples,
        source_name="DAV2",
    )
    return output, {
        "eligible_views": int(len(eligible_image_ids)),
        "views": int(visited_views),
        "accepted_views": int(accepted_views),
        "weak_posterior_views": int(weak_posterior_views),
        "rejected_views": int(rejected_views),
        "trained_rigid_depth_view_count": int(
            len(rigid_depth_maps)
        ),
        "posterior_confidence_percentiles": (
            [
                float(value)
                for value in np.quantile(
                    posterior_confidences,
                    [0.0, 0.1, 0.5, 0.9, 1.0],
                )
            ]
            if posterior_confidences
            else []
        ),
        "posterior_relative_sigma_percentiles": (
            [
                float(value)
                for value in np.quantile(
                    posterior_relative_sigmas,
                    [0.0, 0.1, 0.5, 0.9, 1.0],
                )
            ]
            if posterior_relative_sigmas
            else []
        ),
        **sample_budget_audit,
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


def _line_supported_tree_tracks(
    xyz: np.ndarray,
    tree_instance_id: np.ndarray,
    eligible: np.ndarray,
    *,
    minimum_linearity: float,
    maximum_radial_spread: float = 0.06,
    neighbourhood_radius: float = 0.30,
    minimum_neighbours: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Find cross-view line/cylinder inliers for static trunks and branches.

    Semantic canopy membership or a single-point PCA is not enough evidence
    for a trunk.  A static candidate must already be a cross-sequence
    multi-view track and belong to a same-instance neighbourhood whose two
    minor PCA axes stay compact relative to its long axis.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    instance = np.asarray(tree_instance_id, dtype=np.int32)
    eligible = np.asarray(eligible, dtype=bool)
    supported = np.zeros(len(xyz), dtype=bool)
    local_linearity = np.ones(len(xyz), dtype=np.float32)
    covariance_rows: list[int] = []
    covariances: list[np.ndarray] = []
    for tree in np.unique(instance[eligible]):
        rows = np.flatnonzero(eligible & (instance == tree))
        if len(rows) < int(minimum_neighbours):
            continue
        tree_xyz = xyz[rows]
        neighbours = cKDTree(tree_xyz).query_ball_point(
            tree_xyz, r=float(neighbourhood_radius)
        )
        for local_index, adjacent in enumerate(neighbours):
            if len(adjacent) < int(minimum_neighbours):
                continue
            points = tree_xyz[np.asarray(adjacent, dtype=np.int64)]
            centered = points - points.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(len(points) - 1, 1)
            covariance_rows.append(int(rows[local_index]))
            covariances.append(covariance)
    if covariances:
        # Batched 3x3 eigendecomposition is mathematically identical to one
        # call per candidate, but avoids repeatedly entering a many-threaded
        # BLAS runtime.  The scalar loop made v11 initialization spend tens of
        # minutes after the visual hull had already finished.
        eigenvalues = np.linalg.eigvalsh(np.stack(covariances))
        spread = np.sqrt(np.maximum(eigenvalues, 1e-12))
        linearity = spread[:, 2] / np.maximum(spread[:, 1], 1e-6)
        radial = np.sqrt(
            0.5 * (spread[:, 0] ** 2 + spread[:, 1] ** 2)
        )
        rows = np.asarray(covariance_rows, dtype=np.int64)
        local_linearity[rows] = linearity.astype(np.float32)
        supported[rows] = (
            (linearity >= float(minimum_linearity))
            & (radial <= float(maximum_radial_spread))
            & (spread[:, 2] >= 0.025)
        )
    return supported, local_linearity


def _ownership_isolated_track_frames(
    xyz: np.ndarray,
    dense_ray_local: np.ndarray,
    tree_instance_id: np.ndarray,
    dense_owner_camera_id: np.ndarray,
    minimum: float,
    maximum_cross: float,
    maximum_long: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate frames without crossing geometry or visibility ownership.

    Dense exact-camera births are a sampled optical basis.  Increasing that
    basis must not alter the nearest-neighbour covariance, scale or rotation
    of persistent MASt3R/Chart/DAV2 tracks.  A local covariance must also not
    mix different trees or exact-camera dynamic states: persistent evidence
    is partitioned by physical tree instance, while renderer-basis rows are
    partitioned by ``(owner camera, tree instance)``.  The latter rows are
    never visible outside their owner, so another camera cannot provide a
    valid local shape measurement for them.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    dense = np.asarray(dense_ray_local, dtype=bool).reshape(-1)
    instance = np.asarray(tree_instance_id, dtype=np.int64).reshape(-1)
    owner = np.asarray(dense_owner_camera_id, dtype=np.int64).reshape(-1)
    if not len(xyz) == len(dense) == len(instance) == len(owner):
        raise ValueError("Frame ownership arrays must align with track xyz")
    if bool((owner[dense] < 0).any()):
        raise ValueError("Every dense-ray row needs one exact owner camera")
    scales = np.empty((len(xyz), 3), dtype=np.float32)
    quaternions = np.empty((len(xyz), 4), dtype=np.float32)
    linearity = np.empty(len(xyz), dtype=np.float32)

    def estimate(rows: np.ndarray) -> None:
        if not len(rows):
            return
        if len(rows) == 1:
            scales[rows] = float(minimum)
            quaternions[rows] = np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            linearity[rows] = 1.0
            return
        local_scales, local_quaternions, local_linearity = _track_frames(
            xyz[rows], minimum, maximum_cross, maximum_long
        )
        scales[rows] = local_scales
        quaternions[rows] = local_quaternions
        linearity[rows] = local_linearity

    persistent = np.flatnonzero(~dense)
    persistent_instances = np.unique(instance[persistent])
    for instance_id in persistent_instances:
        estimate(persistent[instance[persistent] == instance_id])

    dense_rows = np.flatnonzero(dense)
    if len(dense_rows):
        dense_keys = np.column_stack(
            [owner[dense_rows], instance[dense_rows]]
        )
        _, dense_group = np.unique(
            dense_keys, axis=0, return_inverse=True
        )
        dense_group_count = int(dense_group.max(initial=-1) + 1)
        for group_id in range(dense_group_count):
            estimate(dense_rows[dense_group == group_id])
    else:
        dense_group_count = 0
    return scales, quaternions, linearity, {
        "contract": (
            "persistent_by_tree_instance__dense_by_exact_owner_camera_and_"
            "tree_instance"
        ),
        "persistent_instance_group_count": int(
            len(persistent_instances)
        ),
        "dense_owner_count": int(len(np.unique(owner[dense_rows]))),
        "dense_owner_instance_group_count": dense_group_count,
        "cross_instance_frame_neighbour_count": 0,
        "cross_owner_frame_neighbour_count": 0,
    }


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
    dense_ray_local = np.asarray(
        [
            bool(point.get("_dense_ray_dynamic_birth", False))
            for point in tracks
        ],
        dtype=bool,
    )
    dense_owner_camera_id = np.full(len(tracks), -1, dtype=np.int64)
    for row in np.flatnonzero(dense_ray_local):
        observation_ids = np.asarray(
            tracks[int(row)].get("observation_camera_ids", ()),
            dtype=np.int64,
        ).reshape(-1)
        observation_ids = observation_ids[observation_ids >= 0]
        if len(observation_ids) != 1:
            raise RuntimeError(
                "A dense renderer-basis row must have exactly one owner "
                "camera"
            )
        dense_owner_camera_id[row] = int(observation_ids[0])
    (
        scales,
        quaternions,
        linearity,
        frame_ownership_audit,
    ) = _ownership_isolated_track_frames(
        xyz,
        dense_ray_local,
        track_instances,
        dense_owner_camera_id,
        0.008,
        0.035,
        0.10,
    )
    ray_footprint = np.asarray(
        [point.get("ray_footprint_scale", 0.0) for point in tracks],
        dtype=np.float32,
    )
    scales[dav2_local] = np.maximum(
        scales[dav2_local],
        ray_footprint[dav2_local, None],
    )
    scales[dense_ray_local] = np.maximum(
        scales[dense_ray_local],
        ray_footprint[dense_ray_local, None],
    )
    track_count = len(tracks)
    support_capacity = max(
        int(hull["support_camera_ids"].shape[1]),
        max(
            (
                len(np.unique(point["tree_image_ids"]))
                for point in tracks
            ),
            default=0,
        ),
    )
    observation_capacity = max(
        int(hull["observation_camera_ids"].shape[1]),
        max(
            (
                min(
                    len(point.get("observation_camera_ids", [])),
                    len(
                        np.asarray(
                            point.get("observation_uv", [])
                        ).reshape(-1, 2)
                    ),
                    len(point.get("observation_depth", [])),
                )
                for point in tracks
            ),
            default=0,
        ),
    )

    def pad_slots(value, width, fill_value):
        value = np.asarray(value)
        if value.shape[1] == int(width):
            return value
        if value.shape[1] > int(width):
            raise RuntimeError("camera metadata width unexpectedly shrank")
        padding = np.full(
            (len(value), int(width) - value.shape[1]) + value.shape[2:],
            fill_value,
            dtype=value.dtype,
        )
        return np.concatenate([value, padding], axis=1)

    # The canonical hull is compacted before this function. Match only the
    # small width actually required by the incoming observation records; do
    # not recreate a 256-slot table for every sequence-local birth.
    hull["support_camera_ids"] = pad_slots(
        hull["support_camera_ids"], support_capacity, -1
    )
    hull["observation_camera_ids"] = pad_slots(
        hull["observation_camera_ids"], observation_capacity, -1
    )
    hull["observation_uv"] = pad_slots(
        hull["observation_uv"], observation_capacity, np.nan
    )
    hull["observation_depth"] = pad_slots(
        hull["observation_depth"], observation_capacity, np.nan
    )
    support_camera_ids = np.full(
        (track_count, support_capacity), -1, dtype=np.int32
    )
    observation_camera_ids = np.full(
        (track_count, observation_capacity), -1, dtype=np.int32
    )
    observation_uv = np.full(
        (track_count, observation_capacity, 2),
        np.nan,
        dtype=np.float32,
    )
    observation_depth = np.full(
        (track_count, observation_capacity), np.nan, dtype=np.float32
    )
    for index, point in enumerate(tracks):
        values = np.unique(point["tree_image_ids"])[:support_capacity]
        support_camera_ids[index, : len(values)] = values
        cameras = np.asarray(
            point.get("observation_camera_ids", []), dtype=np.int32
        )[:observation_capacity]
        uv = np.asarray(
            point.get("observation_uv", []), dtype=np.float32
        ).reshape(-1, 2)[:observation_capacity]
        depth = np.asarray(
            point.get("observation_depth", []), dtype=np.float32
        ).reshape(-1)[:observation_capacity]
        observation_count = min(
            len(cameras), len(uv), len(depth), observation_capacity
        )
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
    measured_position_sigma = np.asarray(
        [
            point.get("position_sigma", np.nan)
            for point in tracks
        ],
        dtype=np.float32,
    )
    measured_sigma_valid = (
        np.isfinite(measured_position_sigma)
        & (measured_position_sigma > 0)
    )
    covariance_scale[measured_sigma_valid] = np.maximum(
        covariance_scale[measured_sigma_valid],
        measured_position_sigma[measured_sigma_valid],
    )
    position_covariance = np.eye(3, dtype=np.float32)[None] * (
        covariance_scale[:, None, None] ** 2
    )
    sequence_support = np.asarray(
        [point["tree_sequence_count"] for point in tracks],
        dtype=np.int16,
    )
    # Cross-sequence tracks supply identity; a same-instance line/cylinder
    # neighbourhood supplies the missing physical trunk/branch evidence.
    trunk_candidate = (
        (sequence_support >= 2)
        & ~chart_local
        & ~dav2_local
        & ~dense_ray_local
        & (error <= maximum_reprojection_error)
    )
    trunk_or_branch_evidence, local_linearity = (
        _line_supported_tree_tracks(
            xyz,
            track_instances,
            trunk_candidate,
            minimum_linearity=skeleton_linearity,
        )
    )
    linearity = np.maximum(linearity, local_linearity)
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
    layer_role[chart_local | dav2_local | dense_ray_local] = 2
    occupancy_probability = np.asarray(
        [point["tree_fraction"] for point in tracks],
        dtype=np.float32,
    )
    support_view_count = np.asarray(
        [len(point["tree_image_ids"]) for point in tracks],
        dtype=np.int16,
    )
    track_ray_depth_nll = np.asarray(
        [
            point.get(
                "ray_depth_nll",
                (
                    1.0
                    / max(
                        float(
                            point.get(
                                "dav2_alignment_posterior", 1.0
                            )
                        ),
                        1.0e-4,
                    )
                    ** 2
                    - 1.0
                    if point.get(
                        "dav2_alignment_posterior"
                    )
                    is not None
                    else 0.0
                ),
            )
            for point in tracks
        ],
        dtype=np.float32,
    )
    track_ray_depth_nll = np.nan_to_num(
        track_ray_depth_nll,
        nan=1.0e8,
        posinf=1.0e8,
        neginf=0.0,
    )
    track_ray_depth_nll = np.maximum(
        track_ray_depth_nll, 0.0
    ).astype(np.float32)
    dense_initial_opacity, _ = evidence_conditioned_leaf_optical_mass(
        occupancy_probability,
        support_view_count,
        np.zeros(track_count, dtype=np.int16),
        track_ray_depth_nll,
        np.zeros(track_count, dtype=np.int16),
    )
    initial_opacity = np.full(track_count, 0.025, dtype=np.float32)
    initial_opacity[dav2_local] = np.clip(
        0.01 + 0.07 * occupancy_probability[dav2_local],
        0.01,
        0.08,
    )
    initial_opacity[dense_ray_local] = (
        dense_initial_opacity[dense_ray_local].numpy()
    )
    values = {
        "centers": xyz,
        "colors": rgb,
        "scales": scales,
        "opacities": initial_opacity[:, None],
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
            np.where(
                dense_ray_local,
                SOURCE_DENSE_RAY,
                np.where(chart_local, SOURCE_CHART, SOURCE_MAST3R),
            ),
        ).astype(np.int8),
        "occupancy_probability": occupancy_probability,
        "position_covariance": position_covariance,
        "reprojection_error": error,
        "track_linearity": linearity,
        "support_camera_ids": support_camera_ids,
        "observation_camera_ids": observation_camera_ids,
        "observation_uv": observation_uv,
        "observation_depth": observation_depth,
        "support_view_count": support_view_count,
        "support_sequence_count": np.asarray(
            [point["tree_sequence_count"] for point in tracks],
            dtype=np.int16,
        ),
        "ray_depth_nll": track_ray_depth_nll,
    }
    merged = {}
    for key, track_value in values.items():
        if key not in hull:
            if key == "observation_camera_ids":
                hull[key] = np.full(
                    (len(hull["centers"]), observation_capacity),
                    -1,
                    dtype=np.int32,
                )
            elif key == "observation_uv":
                hull[key] = np.full(
                    (len(hull["centers"]), observation_capacity, 2),
                    np.nan,
                    dtype=np.float32,
                )
            elif key == "observation_depth":
                hull[key] = np.full(
                    (len(hull["centers"]), observation_capacity),
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
        "line_cylinder_supported_track_count": int(
            trunk_or_branch_evidence.sum()
        ),
        "canonical_track_anchors": int((layer_role == 0).sum()),
        "canonical_crown": int((hull["layer_role"] == 0).sum()),
        "sequence_local_dynamic_tracks": int((layer_role == 2).sum()),
        "dense_ray_dynamic_births": int(dense_ray_local.sum()),
        "dense_dynamic_initial_opacity_quantiles": (
            np.quantile(
                initial_opacity[dense_ray_local],
                [0.0, 0.5, 0.9, 0.99, 1.0],
            ).tolist()
            if bool(dense_ray_local.any())
            else []
        ),
        "dense_dynamic_optical_mass_contract": (
            "exact_owner_support_unknown_free_space_optical_existence_"
            "separate_from_occupancy_depth_geometry_authority"
        ),
        "track_frame_ownership": frame_ownership_audit,
        "tree_instances": int(
            max(
                track_instances.max(),
                hull["tree_instance_id"].max(),
            )
        )
        + 1,
    }


def _local_replacement_groups(
    centers: np.ndarray,
    layer_role: np.ndarray,
    tree_instance_id: np.ndarray,
    *,
    maximum_center_distance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind dynamic leaves only to a physically local canonical crown cell.

    A tree instance is an ownership region, not a replacement cell.  Assigning
    every sequence-local leaf to the nearest canonical point anywhere in the
    same instance allowed metre-scale (and occasionally scene-scale) links.
    Those links let unrelated leaves deform, recolour and retire a canonical
    crown cell at a different depth, which appears as a broad translucent
    smear around both trees and neighbouring facades.

    Canonical rows receive immutable one-row group ids.  A dynamic row joins
    the nearest canonical row in the *same tree instance* only when their
    world-space centres fall within a small local radius.  Tree-instance
    ownership is part of the dynamic deformation, capacity and lifecycle
    contract; crossing it would make one tree's sequence code move or retire
    another tree's crown cell.  Instance fragmentation is repaired by the
    density-supported crown clustering stage, not hidden here.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    layer_role = np.asarray(layer_role, dtype=np.int8).reshape(-1)
    tree_instance_id = np.asarray(
        tree_instance_id, dtype=np.int32
    ).reshape(-1)
    if not (
        len(centers) == len(layer_role) == len(tree_instance_id)
    ):
        raise ValueError(
            "centres, layer roles and tree instances must have equal length"
        )
    maximum_center_distance = float(maximum_center_distance)
    if not np.isfinite(maximum_center_distance) or maximum_center_distance <= 0:
        raise ValueError("maximum_center_distance must be finite and positive")

    groups = np.full(len(centers), -1, dtype=np.int64)
    canonical = np.flatnonzero(layer_role == 0)
    dynamic = np.flatnonzero(layer_role == 2)
    groups[canonical] = np.arange(len(canonical), dtype=np.int64)
    if not len(canonical) or not len(dynamic):
        return groups, {
            "maximum_center_distance": maximum_center_distance,
            "canonical_group_count": int(len(canonical)),
            "dynamic_count": int(len(dynamic)),
            "bound_dynamic_count": 0,
            "unbound_dynamic_count": int(len(dynamic)),
            "bound_canonical_group_count": 0,
            "cross_instance_binding_count": 0,
            "binding_distance_quantiles": [],
        }

    distance = np.full(len(dynamic), np.inf, dtype=np.float64)
    nearest_canonical = np.full(len(dynamic), -1, dtype=np.int64)
    missing_canonical_instances = 0
    for instance_id in np.unique(tree_instance_id[dynamic]):
        dynamic_slots = np.flatnonzero(
            tree_instance_id[dynamic] == instance_id
        )
        canonical_rows = canonical[
            tree_instance_id[canonical] == instance_id
        ]
        if not len(canonical_rows):
            missing_canonical_instances += 1
            continue
        local_distance, local_nearest = cKDTree(
            centers[canonical_rows]
        ).query(
            centers[dynamic[dynamic_slots]],
            k=1,
            workers=-1,
        )
        distance[dynamic_slots] = local_distance
        nearest_canonical[dynamic_slots] = canonical_rows[local_nearest]
    bound = (
        (nearest_canonical >= 0)
        & np.isfinite(distance)
        & (distance <= maximum_center_distance)
    )
    groups[dynamic[bound]] = groups[nearest_canonical[bound]]
    bound_distance = distance[bound]
    return groups, {
        "maximum_center_distance": maximum_center_distance,
        "canonical_group_count": int(len(canonical)),
        "dynamic_count": int(len(dynamic)),
        "bound_dynamic_count": int(bound.sum()),
        "unbound_dynamic_count": int((~bound).sum()),
        "bound_dynamic_fraction": float(bound.mean()),
        "bound_canonical_group_count": int(
            len(np.unique(groups[dynamic[bound]]))
        ),
        "cross_instance_binding_count": 0,
        "dynamic_instance_without_canonical_count": int(
            missing_canonical_instances
        ),
        "binding_distance_quantiles": (
            np.quantile(
                bound_distance, [0.0, 0.5, 0.9, 0.99, 1.0]
            ).tolist()
            if len(bound_distance)
            else []
        ),
    }


def _compact_padded_camera_metadata(
    foliage: dict[str, np.ndarray],
) -> dict[str, int]:
    """Remove padding-only camera slots without changing any observation.

    Visual-hull support is initially accumulated in one column per selected
    view.  With 256 selected views this produced four dense N x 256 metadata
    arrays even though a StMarys primitive has at most 30 support views and
    at most three direct observations.  The renderer ignores every negative
    camera id, so stable row-wise left compaction is representation-equivalent
    while removing several GiB of checkpoint and CUDA state.
    """

    def compact_ids(
        ids: np.ndarray,
        *aligned: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        ids = np.asarray(ids)
        if ids.ndim != 2:
            raise ValueError("padded camera ids must have shape [N, K]")
        row_count, old_width = ids.shape
        for value in aligned:
            if value.shape[:2] != ids.shape:
                raise ValueError(
                    "camera-aligned metadata must share [N, K]"
                )
        valid = ids >= 0
        counts = valid.sum(axis=1, dtype=np.int32)
        new_width = int(counts.max(initial=0))
        compact_ids_array = np.full(
            (row_count, new_width), -1, dtype=ids.dtype
        )
        compact_aligned = [
            np.full(
                (row_count, new_width) + value.shape[2:],
                np.nan,
                dtype=value.dtype,
            )
            for value in aligned
        ]
        cursor = np.zeros(row_count, dtype=np.int32)
        for column in range(old_width):
            rows = np.flatnonzero(valid[:, column])
            if not len(rows):
                continue
            slots = cursor[rows]
            compact_ids_array[rows, slots] = ids[rows, column]
            for destination, source in zip(
                compact_aligned, aligned
            ):
                destination[rows, slots] = source[rows, column]
            cursor[rows] += 1
        if not np.array_equal(cursor, counts):
            raise RuntimeError("camera metadata compaction lost rows")
        return compact_ids_array, compact_aligned, counts

    support = np.asarray(foliage["support_camera_ids"])
    observation = np.asarray(foliage["observation_camera_ids"])
    observation_uv = np.asarray(foliage["observation_uv"])
    observation_depth = np.asarray(foliage["observation_depth"])
    old_support_width = int(support.shape[1])
    old_observation_width = int(observation.shape[1])
    old_bytes = int(
        support.nbytes
        + observation.nbytes
        + observation_uv.nbytes
        + observation_depth.nbytes
    )

    compact_support, _, support_count = compact_ids(support)
    expected_support = np.asarray(
        foliage["support_view_count"], dtype=np.int32
    )
    if not np.array_equal(support_count, expected_support):
        raise RuntimeError(
            "support_view_count disagrees with retained camera ids"
        )
    valid_observation = observation >= 0
    if bool(
        (
            valid_observation
            & (
                ~np.isfinite(observation_uv).all(axis=-1)
                | ~np.isfinite(observation_depth)
            )
        ).any()
    ):
        raise RuntimeError(
            "valid observation camera ids require finite UV and depth"
        )
    (
        compact_observation,
        (compact_uv, compact_depth),
        observation_count,
    ) = compact_ids(
        observation, observation_uv, observation_depth
    )
    foliage["support_camera_ids"] = compact_support
    foliage["observation_camera_ids"] = compact_observation
    foliage["observation_uv"] = compact_uv
    foliage["observation_depth"] = compact_depth
    new_bytes = int(
        compact_support.nbytes
        + compact_observation.nbytes
        + compact_uv.nbytes
        + compact_depth.nbytes
    )
    return {
        "support_slot_width_before": old_support_width,
        "support_slot_width_after": int(compact_support.shape[1]),
        "observation_slot_width_before": old_observation_width,
        "observation_slot_width_after": int(
            compact_observation.shape[1]
        ),
        "support_count_maximum": int(support_count.max(initial=0)),
        "observation_count_maximum": int(
            observation_count.max(initial=0)
        ),
        "camera_metadata_bytes_before": old_bytes,
        "camera_metadata_bytes_after": new_bytes,
        "camera_metadata_bytes_removed": old_bytes - new_bytes,
    }


def _foliage_posterior_view_pool(
    view_records,
    observed_ids,
    selected_view_count,
):
    """Resolve the public ``0 == all fixed cameras`` selection contract."""
    view_records = list(view_records)
    requested = int(selected_view_count)
    if requested <= 0:
        return (
            view_records,
            len(view_records),
            "all_fixed_database_cameras",
        )
    observed_ids = {int(value) for value in observed_ids}
    observed_views = [
        view
        for view in view_records
        if int(view["image_id"]) in observed_ids
    ]
    return (
        observed_views,
        min(len(observed_views), max(24, requested)),
        "bounded_observed_diverse_cameras",
    )


def _select_foliage_posterior_views(
    posterior_view_pool,
    posterior_view_limit,
    posterior_camera_contract,
    *,
    support_by_instance=None,
    minimum_center_distance=0.20,
):
    """Apply diversity only when the public contract requests a subset."""
    posterior_view_pool = list(posterior_view_pool)
    if posterior_camera_contract == "all_fixed_database_cameras":
        if int(posterior_view_limit) != len(posterior_view_pool):
            raise RuntimeError(
                "All-fixed camera contract has a truncated view limit"
            )
        return sorted(
            posterior_view_pool, key=lambda row: int(row["image_id"])
        )
    if support_by_instance:
        return instance_balanced_diverse_views(
            posterior_view_pool,
            support_by_instance=support_by_instance,
            limit=int(posterior_view_limit),
            minimum_center_distance=minimum_center_distance,
        )
    return greedy_diverse_views(
        posterior_view_pool,
        limit=int(posterior_view_limit),
        minimum_center_distance=minimum_center_distance,
    )


def _select_dav2_foliage_views(view_records, selected_view_count):
    """Keep per-camera metric depth aligned with the posterior contract."""
    view_records = list(view_records)
    requested = int(selected_view_count)
    if requested <= 0:
        return (
            sorted(view_records, key=lambda row: int(row["image_id"])),
            "all_fixed_database_cameras",
        )
    limit = min(len(view_records), max(72, requested * 2))
    return (
        sequence_balanced_diverse_views(
            view_records,
            limit=limit,
            minimum_center_distance=0.20,
        ),
        "bounded_sequence_balanced_cameras",
    )


def _balanced_single_camera_sample_cap(
    samples: list[dict[str, Any]],
    maximum_samples: int,
    *,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Apply a global observation budget without dropping later camera ids.

    Gather bounded per-view samples first, then water-fill the global
    capacity equally across every accepted camera. At least one row per
    accepted camera is a hard feasibility requirement: silently keeping a
    camera in provenance while erasing all of its observations would violate
    the posterior contract.
    """
    maximum_samples = int(maximum_samples)
    if maximum_samples < 0:
        raise ValueError("maximum_samples must be non-negative")
    by_camera: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        image_ids = np.asarray(
            sample.get("tree_image_ids", ()), dtype=np.int64
        ).reshape(-1)
        if len(image_ids) != 1:
            raise RuntimeError(
                f"A sequence-local {source_name} sample must own exactly "
                "one camera"
            )
        by_camera.setdefault(int(image_ids[0]), []).append(sample)
    camera_ids = sorted(by_camera)
    raw_count = int(len(samples))
    camera_count = int(len(camera_ids))
    if raw_count <= maximum_samples:
        return list(samples), {
            "raw_samples": raw_count,
            "retained_samples": raw_count,
            "sampled_views": camera_count,
            "camera_budget_truncated": False,
        }
    if maximum_samples < camera_count:
        raise RuntimeError(
            f"{source_name} global sample budget cannot preserve one "
            "observation per "
            f"accepted camera: budget={maximum_samples}, "
            f"cameras={camera_count}"
        )

    quotas = {image_id: 1 for image_id in camera_ids}
    remaining = maximum_samples - camera_count
    while remaining > 0:
        active = [
            image_id
            for image_id in camera_ids
            if quotas[image_id] < len(by_camera[image_id])
        ]
        if not active:
            break
        share = max(1, remaining // len(active))
        assigned = 0
        for image_id in active:
            addition = min(
                share,
                len(by_camera[image_id]) - quotas[image_id],
                remaining - assigned,
            )
            quotas[image_id] += int(addition)
            assigned += int(addition)
            if assigned >= remaining:
                break
        if assigned <= 0:
            raise RuntimeError(
                f"{source_name} balanced budget made no progress"
            )
        remaining -= assigned

    retained = [
        sample
        for image_id in camera_ids
        for sample in by_camera[image_id][: quotas[image_id]]
    ]
    retained_camera_ids = {
        int(np.asarray(row["tree_image_ids"]).reshape(-1)[0])
        for row in retained
    }
    return retained, {
        "raw_samples": raw_count,
        "retained_samples": int(len(retained)),
        "sampled_views": int(len(retained_camera_ids)),
        "camera_budget_truncated": True,
    }


def build_foliage_seed(
    evidence_store: Path,
    surface_seed: Path,
    output: Path,
    *,
    rgb_root: Path | None = None,
    voxel_size: float = 0.12,
    maximum_voxels: int = 400_000,
    selected_view_count: int = 64,
    maximum_dense_rays_per_view: int = 16_384,
    maximum_dense_rays_total: int | None = None,
    minimum_dense_rays_per_view: int = 0,
    maximum_bound_rays_per_view: int = 8192,
    maximum_dynamic_births_per_view: int = 1024,
    maximum_weak_continuous_dynamic_births_per_view: int | None = None,
    dynamic_birth_target_source_pixels_per_basis: float | None = None,
    minimum_track_observations: int = 3,
    minimum_track_sequences: int = 2,
    maximum_reprojection_error: float = 2.0,
    skeleton_linearity: float = 1.8,
    rigid_calibration_ply: Path | None = None,
    rigid_calibration_resolution_scale: float = 0.125,
    seed: int = 73,
) -> dict[str, Any]:
    if int(maximum_dynamic_births_per_view) < 0:
        raise ValueError(
            "maximum_dynamic_births_per_view must be non-negative"
        )
    if (
        maximum_weak_continuous_dynamic_births_per_view is not None
        and int(maximum_weak_continuous_dynamic_births_per_view) < 0
    ):
        raise ValueError(
            "maximum_weak_continuous_dynamic_births_per_view must be "
            "non-negative"
        )
    if (
        dynamic_birth_target_source_pixels_per_basis is not None
        and (
            not np.isfinite(
                float(dynamic_birth_target_source_pixels_per_basis)
            )
            or float(dynamic_birth_target_source_pixels_per_basis) <= 0
        )
    ):
        raise ValueError(
            "dynamic_birth_target_source_pixels_per_basis must be finite "
            "and positive"
        )
    if int(maximum_bound_rays_per_view) < 0:
        raise ValueError(
            "maximum_bound_rays_per_view must be non-negative"
        )
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
    rgb_root = (
        Path(rgb_root).expanduser().resolve()
        if rgb_root is not None
        else (dataset / "images").resolve()
    )
    if not rgb_root.is_dir():
        raise FileNotFoundError(
            f"Foliage target RGB root does not exist: {rgb_root}"
        )
    view_records = [
        _camera_record(image, cameras[image["camera_id"]], masks)
        for image in images.values()
    ]
    rigid_calibration_depth_maps: dict[int, np.ndarray] = {}
    rigid_calibration_audit: dict[str, Any] = {
        "enabled": False,
        "depth_view_count": 0,
        "source": "sparse_surface_seed_zbuffer",
    }
    if rigid_calibration_ply is not None:
        rigid_calibration_ply = (
            Path(rigid_calibration_ply).expanduser().resolve()
        )
        if not rigid_calibration_ply.is_file():
            raise FileNotFoundError(rigid_calibration_ply)
        from outdoor.rigid_occlusion import (
            render_protected_depth_maps,
        )

        rigid_calibration_depth_maps = render_protected_depth_maps(
            view_records,
            structural_ply=rigid_calibration_ply,
            resolution_scale=float(
                rigid_calibration_resolution_scale
            ),
        )
        rigid_calibration_audit = {
            "enabled": True,
            "depth_view_count": int(
                len(rigid_calibration_depth_maps)
            ),
            "source": "trained_native_rigid_2dgs_surface_render",
            "structural_ply": str(rigid_calibration_ply),
            "structural_ply_sha256": sha256_file(
                rigid_calibration_ply
            ),
            "resolution_scale": float(
                rigid_calibration_resolution_scale
            ),
            "role": (
                "dav2_metric_alignment_and_rigid_occlusion_only"
            ),
        }
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
        chart_camera_path = artifact_path(
            store, "chart_cameras", required=False
        )
        if chart_camera_path is not None:
            chart_camera_payload = json.loads(
                chart_camera_path.read_text(encoding="utf-8")
            )
            chart_camera_stems = {
                Path(str(value)).stem
                for value in chart_camera_payload.get("filepaths", ())
            }
        else:
            chart_camera_stems = set()
        pointmap_foliage = _nonchart_pointmap_foliage_samples(
            store,
            images,
            masks,
            exclude_stems=chart_camera_stems,
        )
        chart_foliage, chart_envelope_rejected, envelope_audit = (
            _filter_sequence_local_scene_envelope(
                chart_foliage, rigid_xyz
            )
        )
        pointmap_foliage, pointmap_envelope_rejected, _ = (
            _filter_sequence_local_scene_envelope(
                pointmap_foliage, rigid_xyz
            )
        )
        calibrated_pointmap_foliage = [
            *chart_foliage,
            *pointmap_foliage,
        ]
        # DAV2 is a dense per-frame observation source, not a correspondence
        # graph.  Aligning every near-duplicate video frame both dominates
        # initialization time and lets long traversals overwhelm the
        # posterior.  Reserve multiple sequence-balanced, spatially diverse
        # views per requested hull camera and fit depth only there.
        dav2_views, dav2_camera_contract = (
            _select_dav2_foliage_views(
                view_records, selected_view_count
            )
        )
        dav2_foliage, dav2_audit = _dav2_foliage_samples(
            store,
            images,
            cameras,
            masks,
            rigid_xyz,
            allowed_image_ids={
                int(view["image_id"]) for view in dav2_views
            },
            rigid_depth_maps=rigid_calibration_depth_maps,
        )
        if dav2_camera_contract == "all_fixed_database_cameras":
            expected_dav2_views = int(len(dav2_views))
            if int(dav2_audit.get("views", -1)) != expected_dav2_views:
                raise RuntimeError(
                    "All-fixed DAV2 contract did not visit every selected "
                    f"camera: visited={dav2_audit.get('views')}, "
                    f"expected={expected_dav2_views}"
                )
            if int(dav2_audit.get("sampled_views", -1)) != int(
                dav2_audit.get("accepted_views", -2)
            ):
                raise RuntimeError(
                    "DAV2 sample cap erased all depth observations from an "
                    "accepted fixed camera"
                )
        dav2_foliage, dav2_envelope_rejected, _ = (
            _filter_sequence_local_scene_envelope(
                dav2_foliage, rigid_xyz
            )
        )
        tracks = [
            *tracks,
            *calibrated_pointmap_foliage,
            *dav2_foliage,
        ]
        # Sequence-local Chart pointmaps and metrically aligned DAV2 samples
        # are not correspondence tracks, but they are real ray/depth
        # observations in calibrated cameras. Feed both to the probabilistic
        # hull; a voxel still needs support from different traversals before
        # it can become canonical, so a moving single-sequence leaf cannot be
        # promoted merely because DAV2 observed it once.
        # Metrically aligned DAV2 samples may propose *supplemental* cells, but
        # must not share random budget truncation with MASt3R/Chart anchors.
        # The latter define the stable primary hull; DAV2 receives only the
        # remaining budget and a one-voxel neighbourhood. A supplemental cell
        # still needs compatible support from different traversals, a real
        # baseline and a triangulation angle before becoming canonical.
        hull_candidate_tracks = [
            *hull_tracks,
            *calibrated_pointmap_foliage,
        ]
        hull_tracks = [
            *hull_candidate_tracks,
            *dav2_foliage,
        ]
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
        hull_candidate_tracks = hull_tracks
        cross_sequence_hull = True
        native_track_count = len(tracks)
        chart_foliage = []
        pointmap_foliage = []
        dav2_foliage = []
        dav2_camera_contract = "not_applicable"
        dav2_audit = {
            "eligible_views": 0,
            "views": 0,
            "accepted_views": 0,
            "rejected_views": 0,
        }
        chart_envelope_rejected = 0
        pointmap_envelope_rejected = 0
        dav2_envelope_rejected = 0
        envelope_audit = {
            "reference_radius": 0.0,
            "maximum_radius": 0.0,
        }
    if not tracks:
        raise RuntimeError("No multi-view semantic tree tracks survived")
    track_xyz = np.stack([point["xyz"] for point in tracks])
    if (
        mast3r_only
        and (chart_foliage or pointmap_foliage)
        and native_track_count
    ):
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
    if mast3r_only:
        hull_xyz = np.stack([point["xyz"] for point in hull_tracks])
        hull_instances = instances[
            cKDTree(track_xyz).query(hull_xyz, k=1)[1]
        ]
        support_by_instance = {}
        for instance, track in zip(hull_instances, hull_tracks):
            row = support_by_instance.setdefault(int(instance), {})
            for image_id in track.get("tree_image_ids", ()):
                image_id = int(image_id)
                row[image_id] = row.get(image_id, 0) + 1
        # A canonical instance must be observed by at least two cameras from
        # two traversals.  Isolated sequence-local leaves belong to the
        # dynamic branch and must not consume the global posterior schedule.
        support_by_instance = {
            instance: row
            for instance, row in support_by_instance.items()
            if len(row) >= 2
            and len(
                {
                    sequence_id(images[image_id]["name"])
                    for image_id in row
                }
            )
            >= 2
        }
        if not support_by_instance:
            fallback = {}
            for track in hull_tracks:
                for image_id in track.get("tree_image_ids", ()):
                    image_id = int(image_id)
                    fallback[image_id] = fallback.get(image_id, 0) + 1
            support_by_instance = {-1: fallback}
        observed_ids = {
            image_id
            for row in support_by_instance.values()
            for image_id in row
        }
        # ``selected_view_count`` is the quality/cost contract for the
        # posterior itself.  The former extra division by two reduced a
        # requested 96-view hull to 48 cameras; after component/depth
        # compatibility only 15 cameras produced interval rays.  In this
        # scene there are just 66 cameras with any retained foliage
        # observation, so consume all of them rather than dropping dense
        # DAV2/Chart views before the probabilistic hull is even built.
        (
            posterior_view_pool,
            posterior_view_limit,
            posterior_camera_contract,
        ) = _foliage_posterior_view_pool(
            view_records,
            observed_ids,
            selected_view_count,
        )
        selected = _select_foliage_posterior_views(
            posterior_view_pool,
            posterior_view_limit,
            posterior_camera_contract,
            support_by_instance=support_by_instance,
            minimum_center_distance=0.20,
        )
        selected_ids = {int(view["image_id"]) for view in selected}
        selected_instance_support = {
            str(instance): {
                "available_camera_count": len(camera_counts),
                "selected_camera_count": len(
                    selected_ids & set(camera_counts)
                ),
                "selected_sequence_count": len(
                    {
                        sequence_id(images[image_id]["name"])
                        for image_id in selected_ids & set(camera_counts)
                    }
                ),
            }
            for instance, camera_counts in sorted(
                support_by_instance.items()
            )
        }
    else:
        (
            posterior_view_pool,
            posterior_view_limit,
            posterior_camera_contract,
        ) = _foliage_posterior_view_pool(
            view_records,
            {int(view["image_id"]) for view in view_records},
            selected_view_count,
        )
        selected = _select_foliage_posterior_views(
            posterior_view_pool,
            posterior_view_limit,
            posterior_camera_contract,
            minimum_center_distance=0.75,
        )
        selected_instance_support = {}
    # Bind every observation-space birth to the same immutable RGB target
    # raster that the teacher will optimize. Source camera coordinates are
    # converted to the target raster inside the dense posterior, so this also
    # closes the 1920x1080 evidence -> 640x360 training-resolution contract.
    for view in selected:
        target_rgb_path = rgb_root / str(view["image_name"])
        if not target_rgb_path.is_file():
            raise FileNotFoundError(
                "Selected foliage view has no contracted target RGB: "
                f"{target_rgb_path}"
            )
        view["target_rgb_path"] = str(target_rgb_path)
    if rigid_calibration_depth_maps:
        rigid_depth = {
            int(view["image_id"]): rigid_calibration_depth_maps[
                int(view["image_id"])
            ]
            for view in selected
            if int(view["image_id"])
            in rigid_calibration_depth_maps
        }
    else:
        rigid_depth = sparse_rigid_depth_maps(selected, rigid_xyz)
    hull = build_instance_aware_canopy_volume(
        hull_tracks,
        images,
        cameras,
        masks,
        candidate_tracks=hull_candidate_tracks,
        supplemental_candidate_tracks=dav2_foliage,
        rigid_depth_maps=rigid_depth,
        selected_views=selected,
        minimum_support_sequences=2,
        voxel_size=voxel_size,
        maximum_voxels=maximum_voxels,
        maximum_dense_rays_per_view=maximum_dense_rays_per_view,
        maximum_dense_rays_total=maximum_dense_rays_total,
        minimum_dense_rays_per_view=minimum_dense_rays_per_view,
        maximum_bound_rays_per_view=maximum_bound_rays_per_view,
        maximum_dynamic_births_per_view=(
            maximum_dynamic_births_per_view
        ),
        maximum_weak_continuous_dynamic_births_per_view=(
            maximum_weak_continuous_dynamic_births_per_view
        ),
        dynamic_birth_target_source_pixels_per_basis=(
            dynamic_birth_target_source_pixels_per_basis
        ),
        # The hull builder now requires cross-sequence depth hits, coherent
        # local line/cylinder geometry, wood-compatible RGB, zero confirmed
        # free-space contradictions and bounded ray NLL. This is measured
        # trunk/branch evidence rather than the removed whole-crown PCA axis.
        allow_inferred_skeleton=True,
        seed=seed,
    )
    # Dense observation-space rays were previously retained only in the
    # posterior table.  If their single-view hit could not pass the canonical
    # cross-sequence hull gate, no renderer primitive ever consumed it:
    # volume split preserves an existing footprint and cannot fill an
    # ownerless silhouette hole.  Keep every such hit as an explicitly
    # sequence-local dynamic birth with a unique lineage.  The local hull
    # alignment below assigns physical ownership without promoting it to the
    # canonical localization map.
    persistent_track_count = len(tracks)
    dense_dynamic_births = list(
        hull.pop("dense_dynamic_births", ())
    )
    dense_dynamic_birth_count = int(
        hull.pop(
            "dense_dynamic_birth_count",
            len(dense_dynamic_births),
        )
    )
    if dense_dynamic_birth_count != len(dense_dynamic_births):
        raise RuntimeError(
            "Dense dynamic-birth audit count does not match its records"
        )
    dense_dynamic_rgb_sources = {
        str(point.get("_dense_ray_rgb_source", "missing"))
        for point in dense_dynamic_births
    }
    if dense_dynamic_births and dense_dynamic_rgb_sources != {
        "exact_training_target_raster"
    }:
        raise RuntimeError(
            "Production dense dynamic births were not initialized from the "
            f"contracted RGB target: {sorted(dense_dynamic_rgb_sources)}"
        )
    if dense_dynamic_births:
        existing_ids = {int(point["id"]) for point in tracks}
        next_id = -50_000_000
        for point in dense_dynamic_births:
            while next_id in existing_ids:
                next_id -= 1
            point["id"] = next_id
            existing_ids.add(next_id)
            next_id -= 1
        tracks.extend(dense_dynamic_births)
        track_xyz = np.stack([point["xyz"] for point in tracks])
    # The ray/depth-supported hull has the stronger ownership evidence. Keep
    # its local components authoritative and associate only nearby
    # sequence-local observations. Mapping in the opposite direction to five
    # sparse native-track ids previously recreated two 144 m global
    # instances, defeating balanced topology and local replacement.
    hull_component_count = int(hull["tree_instance_count"])
    if len(hull["centers"]):
        (
            hull["tree_instance_id"],
            instances,
            instance_alignment_audit,
        ) = _align_tracks_to_local_hull_instances(
            np.asarray(hull["centers"], dtype=np.float64),
            np.asarray(hull["tree_instance_id"], dtype=np.int32),
            track_xyz,
            association_radius=max(1.5, 8.0 * float(voxel_size)),
            maximum_component_extent=12.0,
            persistent_track_count=persistent_track_count,
        )
        shared_hull_instance_count = int(
            len(np.unique(hull["tree_instance_id"]))
        )
        instance_alignment_distance_median = float(
            instance_alignment_audit["association_distance_median"]
        )
        instance_alignment_distance_p99 = float(
            instance_alignment_audit["association_distance_p99"]
        )
    else:
        instance_alignment_audit = {}
        shared_hull_instance_count = 0
        instance_alignment_distance_median = float("nan")
        instance_alignment_distance_p99 = float("nan")
    selected_views = hull.pop("selected_views")
    candidate_count = int(hull.pop("candidate_count"))
    dense_observation_ray_count = int(
        hull.pop("dense_observation_ray_count", 0)
    )
    dense_hole_proposal_count = int(
        hull.pop("dense_hole_proposal_count", 0)
    )
    dense_ray_budget = dict(hull.pop("dense_ray_budget"))
    candidate_bound_ray_budget = dict(
        hull.pop("candidate_bound_ray_budget")
    )
    rigid_depth_view_count = int(hull.pop("rigid_depth_view_count"))
    hull.pop("tree_instance_count")
    ray_evidence = hull.pop("ray_evidence")
    # Compact the canonical rows before adding almost one million
    # sequence-local births.  The former implementation first expanded every
    # birth to the selected-view width (256), filled one or three slots, and
    # only then removed the padding.  Besides wasting ~5 GB, that serial fill
    # dominated from-zero initialization.
    legacy_hull_slot_width = int(hull["support_camera_ids"].shape[1])
    legacy_hull_row_count = int(len(hull["centers"]))
    if "observation_camera_ids" not in hull:
        # Canonical visual-hull rows are supported by persistent ray
        # intervals, not a single point observation.  State that as a
        # zero-width table now; _merge_foliage formerly fabricated 256 empty
        # slots per row as an incidental side effect.
        hull["observation_camera_ids"] = np.empty(
            (legacy_hull_row_count, 0), dtype=np.int32
        )
        hull["observation_uv"] = np.empty(
            (legacy_hull_row_count, 0, 2), dtype=np.float32
        )
        hull["observation_depth"] = np.empty(
            (legacy_hull_row_count, 0), dtype=np.float32
        )
    hull_camera_metadata_storage = _compact_padded_camera_metadata(hull)
    track_count_for_metadata = len(tracks)
    legacy_track_metadata_bytes = int(
        track_count_for_metadata
        * (
            20 * legacy_hull_slot_width
        )
    )
    legacy_hull_observation_bytes = int(
        legacy_hull_row_count * 16 * legacy_hull_slot_width
    )
    merged, counts = _merge_foliage(
        hull,
        tracks,
        instances,
        skeleton_linearity=skeleton_linearity,
        maximum_reprojection_error=maximum_reprojection_error,
    )
    final_camera_metadata_storage = _compact_padded_camera_metadata(merged)
    logical_metadata_bytes_before = int(
        hull_camera_metadata_storage["camera_metadata_bytes_before"]
        + legacy_hull_observation_bytes
        + legacy_track_metadata_bytes
    )
    camera_metadata_storage = {
        **final_camera_metadata_storage,
        "support_slot_width_before": int(
            hull_camera_metadata_storage["support_slot_width_before"]
        ),
        "observation_slot_width_before": int(
            legacy_hull_slot_width
        ),
        "camera_metadata_bytes_before": logical_metadata_bytes_before,
        "camera_metadata_bytes_removed": int(
            logical_metadata_bytes_before
            - final_camera_metadata_storage[
                "camera_metadata_bytes_after"
            ]
        ),
        "construction_contract": (
            "compact_hull_then_direct_width_sequence_local_rows"
        ),
        "premerge_hull_compaction": hull_camera_metadata_storage,
        "legacy_padded_track_metadata_bytes_avoided": (
            legacy_track_metadata_bytes
        ),
        "legacy_padded_hull_observation_bytes_avoided": (
            legacy_hull_observation_bytes
        ),
    }
    # A dense observation-space birth represents one finite source-image
    # footprint.  Persist a local bandwidth ceiling instead of allowing the
    # optimizer to grow it back to the scene-wide 12 cm cap after every
    # split.  Non-dense tracks retain the normal global role cap.
    scale_ceiling = np.full_like(
        merged["scales"], np.inf, dtype=np.float32
    )
    dense_ray = merged["initialization_source"] == SOURCE_DENSE_RAY
    scale_ceiling[dense_ray] = (
        merged["scales"][dense_ray] * 1.10
    ).astype(np.float32)
    merged["scale_ceiling"] = scale_ceiling
    (
        merged["replacement_group"],
        replacement_group_audit,
    ) = _local_replacement_groups(
        merged["centers"],
        merged["layer_role"],
        merged["tree_instance_id"],
        # A canonical voxel and an observation-local leaf may differ by more
        # than one 12 cm cell after posterior interpolation, but a replacement
        # link spanning more than three cells is no longer a local ownership
        # relation.
        maximum_center_distance=max(0.30, 3.0 * float(voxel_size)),
    )
    # Exact owner RGB is not independent metric-depth evidence, but its tree
    # mask *is* independent evidence that optical mass exists somewhere on
    # that calibrated ray.  Keep the spatial geometry-authority value as an
    # audit diagnostic; do not project optical alpha onto it.  Ray factors,
    # topology and the training floor use this uncertainty continuously.
    dynamic = merged["layer_role"] == 2
    dynamic_ceiling = (
        evidence_conditioned_dynamic_opacity_ceiling(
            merged["occupancy_probability"],
            merged["support_view_count"],
            merged["unknown_view_count"],
            merged["ray_depth_nll"],
            merged["free_space_violation_count"],
            merged["replacement_group"],
            maximum_opacity=0.40,
        )
        .numpy()
        .astype(np.float32)
    )
    dense_dynamic = dynamic & (
        merged["initialization_source"] == SOURCE_DENSE_RAY
    )
    ownerless_dense_dynamic = dense_dynamic & (
        merged["replacement_group"] < 0
    )
    dynamic_authority_audit = {
        "contract": (
            "diagnostic_continuous_depth_reliability_times_local_"
            "replacement_or_ownerless_occupancy__not_an_alpha_gate"
        ),
        "dynamic_count": int(dynamic.sum()),
        "dense_dynamic_count": int(dense_dynamic.sum()),
        "ownerless_dense_dynamic_count": int(
            ownerless_dense_dynamic.sum()
        ),
        "initial_opacity_rows_reduced": 0,
        "dense_ray_depth_reliability_quantiles": (
            np.quantile(
                1.0
                / np.sqrt(
                    1.0
                    + merged["ray_depth_nll"][dense_dynamic]
                ),
                [0.0, 0.1, 0.5, 0.9, 1.0],
            ).tolist()
            if bool(dense_dynamic.any())
            else []
        ),
        "ownerless_dense_geometry_authority_prior_quantiles": (
            np.quantile(
                dynamic_ceiling[ownerless_dense_dynamic],
                [0.0, 0.1, 0.5, 0.9, 1.0],
            ).tolist()
            if bool(ownerless_dense_dynamic.any())
            else []
        ),
    }
    counts["dense_dynamic_initial_opacity_quantiles"] = (
        np.quantile(
            merged["opacities"][dense_dynamic, 0],
            [0.0, 0.5, 0.9, 0.99, 1.0],
        ).tolist()
        if bool(dense_dynamic.any())
        else []
    )
    counts["dense_dynamic_optical_mass_contract"] = (
        "exact_owner_pixel_optical_existence_prior_separate_from_spatial_"
        "depth_geometry_authority__rgb_can_reduce_alpha__"
        "no_hard_per_candidate_alpha_gate"
    )
    merged_instance_extents = []
    for instance_id in np.unique(merged["tree_instance_id"]):
        rows = merged["centers"][
            merged["tree_instance_id"] == instance_id
        ]
        merged_instance_extents.append(
            float(np.linalg.norm(rows.max(axis=0) - rows.min(axis=0)))
        )
    instance_extent_audit = {
        "instance_count": int(len(merged_instance_extents)),
        "bbox_diagonal_median": (
            float(np.median(merged_instance_extents))
            if merged_instance_extents
            else 0.0
        ),
        "bbox_diagonal_p99": (
            float(np.quantile(merged_instance_extents, 0.99))
            if merged_instance_extents
            else 0.0
        ),
        "bbox_diagonal_maximum": (
            float(np.max(merged_instance_extents))
            if merged_instance_extents
            else 0.0
        ),
    }
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "geometry_version": (
            "unified_evidence_dense_ray_bounded_instances_v11_exact_owner_"
            "instance_isolated_bandwidth"
        ),
        "audit": {
            "protocol": INITIALIZATION_VERSION,
            "evidence_hash": store["evidence_hash"],
            "candidate_voxels": candidate_count,
            "dense_observation_ray_count": (
                dense_observation_ray_count
            ),
            "dense_hole_proposal_count": dense_hole_proposal_count,
            "dense_dynamic_birth_count": dense_dynamic_birth_count,
            "ray_to_dynamic_basis_ratio": float(
                dense_observation_ray_count
                / max(dense_dynamic_birth_count, 1)
            ),
            "ray_posterior_renderer_basis_decoupled": bool(
                dense_observation_ray_count
                > dense_dynamic_birth_count
            ),
            "persistent_track_frame_contract": (
                "persistent_knn_partitioned_by_tree_instance__dense_knn_"
                "partitioned_by_exact_owner_camera_and_tree_instance"
            ),
            "maximum_dense_rays_per_view": int(
                maximum_dense_rays_per_view
            ),
            "maximum_dense_rays_total": (
                None
                if maximum_dense_rays_total is None
                else int(maximum_dense_rays_total)
            ),
            "minimum_dense_rays_per_view": int(
                minimum_dense_rays_per_view
            ),
            "maximum_bound_rays_per_view": int(
                maximum_bound_rays_per_view
            ),
            "maximum_dynamic_births_per_view": int(
                maximum_dynamic_births_per_view
            ),
            "maximum_weak_continuous_dynamic_births_per_view": (
                None
                if maximum_weak_continuous_dynamic_births_per_view is None
                else int(
                    maximum_weak_continuous_dynamic_births_per_view
                )
            ),
            "dynamic_birth_target_source_pixels_per_basis": (
                None
                if dynamic_birth_target_source_pixels_per_basis is None
                else float(
                    dynamic_birth_target_source_pixels_per_basis
                )
            ),
            "dense_ray_budget": dense_ray_budget,
            "candidate_bound_ray_budget": (
                candidate_bound_ray_budget
            ),
            "camera_metadata_storage": camera_metadata_storage,
            "dense_ray_footprint_ceiling": (
                "1.10x_initial_then_inherited_and_shrunk_by_split"
            ),
            "dense_dynamic_basis_contract": (
                "instance_balanced_half_two_dimensional_lattice_coverage_"
                "half_spatial_high_frequency_local_rgb_residual_over_"
                "shared_canonical_crown"
            ),
            "dense_ownerless_hit_has_dynamic_consumer": (
                "not_guaranteed__spatially_stratified_exact_camera_basis_"
                "plus_runtime_verified_canonical_fallback"
            ),
            "dense_ownerless_dynamic_consumer_guaranteed": False,
            "dense_ownerless_consumer_coverage_audit": (
                "foliage_ray_runtime_exact_interval_mass"
            ),
            "dense_dynamic_rgb_source": (
                "exact_training_target_raster"
                if dense_dynamic_births
                else "not_applicable"
            ),
            "dense_dynamic_rgb_root": str(rgb_root),
            "ray_posterior_candidate_independent": True,
            "sequence_local_scene_envelope": {
                **envelope_audit,
                "chart_rejected": int(chart_envelope_rejected),
                "nonchart_pointmap_rejected": int(
                    pointmap_envelope_rejected
                ),
                "dav2_rejected": int(dav2_envelope_rejected),
            },
            "rigid_depth_view_count": rigid_depth_view_count,
            "tree_instance_count": int(len(np.unique(instances))),
            "visual_hull_component_count_before_alignment": (
                hull_component_count
            ),
            "shared_visual_hull_instance_count": (
                shared_hull_instance_count
            ),
            "instance_alignment_distance_median": (
                instance_alignment_distance_median
            ),
            "instance_alignment_distance_p99": (
                instance_alignment_distance_p99
            ),
            "instance_alignment": instance_alignment_audit,
            "merged_instance_extent": instance_extent_audit,
            "local_replacement_groups": replacement_group_audit,
            "dynamic_geometry_authority": dynamic_authority_audit,
            "ungrouped_dynamic_visibility": "exact_owner_only",
            "positive_negative_unknown_evidence": True,
            "real_ray_depth_posterior": True,
            "sparse_rigid_occlusion_zbuffer": not bool(
                rigid_calibration_depth_maps
            ),
            "trained_rigid_occlusion_zbuffer": bool(
                rigid_calibration_depth_maps
            ),
            "inferred_visual_hull_skeleton_used": False,
            "static_skeleton_evidence": (
                "cross_sequence_min3_observation_depth_rgb__"
                "same_instance_min3_local_line_cylinder_v2"
            ),
            "historical_trained_ply_used": False,
            "geometry_source": store.get(
                "geometry_source", "legacy_mixed"
            ),
            "colmap_points_or_tracks_used": not mast3r_only,
            **counts,
            "posterior_camera_contract": (
                posterior_camera_contract
            ),
            "dav2_camera_contract": dav2_camera_contract,
            "rigid_depth_calibration": rigid_calibration_audit,
            "selected_views": [
                {
                    "image_id": int(view["image_id"]),
                    "image_name": str(view["image_name"]),
                    "sequence_id": str(view["sequence_id"]),
                }
                for view in selected_views
            ],
            "selected_instance_support": selected_instance_support,
        },
        **{
            key: torch.from_numpy(value)
            for key, value in merged.items()
        },
        "ray_evidence": {
            key: torch.from_numpy(value)
            for key, value in ray_evidence.items()
        }
        | {"depth_coordinate": "camera_z"},
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
        "nonchart_pointmap_sequence_local_sample_count": int(
            len(pointmap_foliage)
        ),
        "all_pointmap_sequence_local_sample_count": int(
            len(chart_foliage) + len(pointmap_foliage)
        ),
        "dav2_sequence_local_sample_count": int(len(dav2_foliage)),
        "dav2_depth_alignment": dav2_audit,
        "visual_hull_candidate_anchor_count": int(
            len(hull_candidate_tracks)
        ),
        "visual_hull_observation_anchor_count": int(len(hull_tracks)),
        "cross_sequence_visual_hull": bool(
            len({view["sequence_id"] for view in selected_views}) >= 2
        ),
        "localization_landmark_eligible": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
