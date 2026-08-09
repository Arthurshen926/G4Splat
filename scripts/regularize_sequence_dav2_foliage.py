#!/usr/bin/env python3
"""Repair per-frame DAV2 scale breaks with fixed-camera reprojection.

The operation is deliberately an initialization transform.  It changes no
RGB, camera, MAtCha/MASt3R, SfM, or trained model state.  DAV2-owned foliage
centres, covariances, projected footprints and their exact positive ray
intervals are moved together, so training never sees a primitive/ray
contradiction after the metric repair.
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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2d-gaussian-splatting"))

from outdoor.evidence_store import load_evidence_store
from outdoor.foliage_geometry import (
    enforce_strict_foliage_ray_intervals,
    quaternion_to_rotation,
    read_cameras_binary,
    read_images_binary,
)
from outdoor.sequence_depth import (
    SEQUENCE_METRIC_DEPTH_VERSION,
    SequenceDepthCamera,
    SequenceDepthView,
    solve_sequence_depth_scales,
)


PROTOCOL = SEQUENCE_METRIC_DEPTH_VERSION
FRAME_PATTERN = re.compile(r"^(?P<sequence>.+)__frame(?P<frame>[0-9]+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_intrinsics(camera: dict) -> tuple[float, float, float, float]:
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = float(camera["params"][0])
        cx, cy = map(float, camera["params"][1:3])
    else:
        fx, fy, cx, cy = map(float, camera["params"][:4])
    return fx, fy, cx, cy


def _group_rows(camera_ids: np.ndarray, selected: np.ndarray) -> dict[int, np.ndarray]:
    indices = np.flatnonzero(selected)
    if not len(indices):
        return {}
    order = np.argsort(camera_ids[indices], kind="stable")
    indices = indices[order]
    values = camera_ids[indices]
    return {
        int(camera_id): indices[values == camera_id]
        for camera_id in np.unique(values)
    }


def _owner_camera_and_uv(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = payload["observation_camera_ids"]
    valid = observations >= 0
    slots = valid.to(torch.int64).argmax(dim=1)
    rows = torch.arange(len(observations))
    owners = observations[rows, slots]
    uv = payload["observation_uv"][rows, slots]
    counts = valid.sum(dim=1)
    return owners.numpy(), uv.numpy(), counts.numpy()


def _build_views(
    payload: dict,
    manifest: dict,
    images: dict[int, dict],
    cameras: dict[int, dict],
) -> tuple[
    dict[str, list[SequenceDepthView]],
    dict[int, SequenceDepthCamera],
    dict[int, np.ndarray],
    dict[str, object],
]:
    source = payload["initialization_source"].numpy()
    error = payload["reprojection_error"].numpy()
    owner, normalized_uv, observation_counts = _owner_camera_and_uv(payload)
    dav2 = (source == 3) | (source == 4)
    source3 = source == 3
    source4 = source == 4
    all_rows = _group_rows(owner, dav2)
    source3_rows = _group_rows(owner, source3)
    source4_rows = _group_rows(owner, source4)

    by_sequence: dict[str, list[SequenceDepthView]] = {}
    camera_lookup: dict[int, SequenceDepthCamera] = {}
    exact_rows: dict[int, np.ndarray] = {}
    missing_scene_camera = 0
    insufficient_births = 0
    multi_observation_rows = int(
        ((observation_counts > 1) & dav2).sum()
    )
    fixed = manifest["foliage"].get(
        "fixed_camera_sequences",
        payload.get("audit", {}).get("fixed_camera_sequences", []),
    )
    for record in fixed:
        camera_id = int(record["image_id"])
        name = Path(str(record["image_name"])).stem
        parsed = FRAME_PATTERN.match(name)
        image = images.get(camera_id)
        if parsed is None or image is None:
            missing_scene_camera += 1
            continue
        camera_record = cameras.get(int(image["camera_id"]))
        if camera_record is None:
            missing_scene_camera += 1
            continue
        preferred = source4_rows.get(camera_id, np.empty(0, dtype=np.int64))
        if len(preferred) < 256:
            temporal = source3_rows.get(
                camera_id, np.empty(0, dtype=np.int64)
            )
            if len(temporal) >= 256:
                preferred = temporal
        if len(preferred) < 64:
            preferred = all_rows.get(
                camera_id, np.empty(0, dtype=np.int64)
            )
        if len(preferred) < 64:
            insufficient_births += 1
            continue

        rotation = quaternion_to_rotation(image["qvec"])
        translation = np.asarray(image["tvec"], dtype=np.float64)
        center = (-translation) @ rotation
        fx, fy, cx, cy = _camera_intrinsics(camera_record)
        camera = SequenceDepthCamera(
            camera_id=camera_id,
            sequence=parsed.group("sequence"),
            frame=int(parsed.group("frame")),
            rotation=np.asarray(rotation, dtype=np.float64),
            translation=translation,
            center=center,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=int(camera_record["width"]),
            height=int(camera_record["height"]),
        )
        independent = source3_rows.get(
            camera_id, np.empty(0, dtype=np.int64)
        )
        # Added temporal witnesses carry a deliberately large uncertainty
        # marker.  The lower error quantile recovers the independent rigid-fit
        # posterior when both kinds coexist, while all-temporal cameras retain
        # a correspondingly weak prior without a binary status check.
        relative_rmse = (
            float(np.quantile(error[independent], 0.1))
            if len(independent)
            else 10.0
        )
        view = SequenceDepthView(
            camera=camera,
            points=payload["centers"][preferred].numpy(),
            normalized_uv=normalized_uv[preferred],
            relative_rmse=relative_rmse,
        )
        by_sequence.setdefault(camera.sequence, []).append(view)
        camera_lookup[camera_id] = camera
        if len(source4_rows.get(camera_id, ())) >= 256:
            exact_rows[camera_id] = source4_rows[camera_id]
        elif len(source3_rows.get(camera_id, ())) >= 256:
            exact_rows[camera_id] = source3_rows[camera_id]

    for sequence in by_sequence:
        by_sequence[sequence].sort(key=lambda value: value.camera.frame)
    audit = {
        "fixed_camera_records": len(fixed),
        "sequence_count": len(by_sequence),
        "graph_camera_count": sum(map(len, by_sequence.values())),
        "exact_ray_camera_count": len(exact_rows),
        "missing_scene_camera_records": missing_scene_camera,
        "insufficient_birth_camera_records": insufficient_births,
        "multi_observation_dav2_rows": multi_observation_rows,
        "owner_contract": (
            "single_exact_camera_for_dav2_rows__multi_observation_rows_are_"
            "audited_and_left_unchanged"
        ),
    }
    return by_sequence, camera_lookup, exact_rows, audit


def _solve_all_sequences(
    by_sequence: dict[str, list[SequenceDepthView]],
    *,
    maximum_frame_gap: int,
    maximum_pixel_distance: float,
    minimum_matches: int,
    iterations: int,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    scales: dict[int, float] = {}
    audits = []
    for sequence in sorted(by_sequence):
        result, audit = solve_sequence_depth_scales(
            by_sequence[sequence],
            maximum_frame_gap=maximum_frame_gap,
            maximum_pixel_distance=maximum_pixel_distance,
            minimum_matches=minimum_matches,
            iterations=iterations,
        )
        scales.update(result)
        audits.append(audit)
    return scales, audits


def _apply_primitive_scales(
    payload: dict,
    scales: dict[int, float],
    camera_lookup: dict[int, SequenceDepthCamera],
) -> tuple[np.ndarray, dict[str, object]]:
    source = payload["initialization_source"].numpy()
    owner, _, observation_counts = _owner_camera_and_uv(payload)
    dav2 = (source == 3) | (source == 4)
    changed: list[np.ndarray] = []
    old_centers = payload["centers"]
    for camera_id, scale in scales.items():
        rows = np.flatnonzero(
            dav2 & (owner == int(camera_id)) & (observation_counts == 1)
        )
        if not len(rows):
            continue
        index = torch.from_numpy(rows)
        center = torch.as_tensor(
            camera_lookup[camera_id].center,
            dtype=old_centers.dtype,
        )
        value = float(scale)
        old_centers[index] = center + value * (old_centers[index] - center)
        payload["scales"][index] *= value
        payload["position_covariance"][index] *= value * value
        if "scale_ceiling" in payload:
            ceiling = payload["scale_ceiling"]
            ceiling[index] *= value
        observations = payload["observation_camera_ids"][index]
        # Advanced indexing returns a copy.  Mutating only that temporary
        # left the Gaussian centre and exact ray interval at the repaired
        # depth while its primitive posterior still reported the old value.
        depth = payload["observation_depth"][index].clone()
        depth[observations == int(camera_id)] *= value
        payload["observation_depth"][index] = depth
        changed.append(rows)
    changed_rows = (
        np.concatenate(changed) if changed else np.empty(0, dtype=np.int64)
    )
    audit = {
        "changed_primitive_rows": int(len(changed_rows)),
        "unchanged_multi_observation_dav2_rows": int(
            (dav2 & (observation_counts > 1)).sum()
        ),
        "joint_parameter_update": (
            "center_observation_depth_covariance_tangent_scales_and_ceiling"
        ),
        "opacity_changed": False,
        "rgb_changed": False,
    }
    return changed_rows, audit


def _rebind_local_ownership(
    payload: dict,
    changed_rows: np.ndarray,
    *,
    maximum_distance: float,
) -> dict[str, object]:
    canonical = np.flatnonzero(
        (payload["layer_role"].numpy() < 2)
        & (payload["replacement_group"].numpy() >= 0)
    )
    if not len(canonical) or not len(changed_rows):
        return {"changed_rows": 0, "supported_rows": 0}
    tree = cKDTree(payload["centers"][canonical].numpy())
    supported = 0
    rebound = 0
    distances = []
    for start in range(0, len(changed_rows), 100_000):
        rows = changed_rows[start : start + 100_000]
        distance, nearest = tree.query(payload["centers"][rows].numpy(), k=1)
        valid = np.isfinite(distance) & (distance <= float(maximum_distance))
        if not bool(valid.any()):
            continue
        local_rows = rows[valid]
        owner_rows = canonical[nearest[valid]]
        previous = payload["replacement_group"][local_rows].clone()
        payload["replacement_group"][local_rows] = payload[
            "replacement_group"
        ][owner_rows]
        payload["tree_instance_id"][local_rows] = payload[
            "tree_instance_id"
        ][owner_rows]
        supported += int(valid.sum())
        rebound += int(
            (previous != payload["replacement_group"][local_rows]).sum()
        )
        distances.extend(distance[valid].tolist())
    return {
        "changed_rows": rebound,
        "supported_rows": supported,
        "maximum_canonical_distance": float(maximum_distance),
        "distance_quantiles": (
            np.quantile(distances, [0.0, 0.5, 0.9, 0.99, 1.0]).tolist()
            if distances
            else []
        ),
        "contract": (
            "nearest_persistent_envelope_after_metric_reprojection__"
            "unsupported_rows_retain_ownerless_or_previous_identity"
        ),
    }


def _apply_exact_ray_scales(
    payload: dict,
    scales: dict[int, float],
    exact_rows: dict[int, np.ndarray],
    *,
    maximum_pixel_distance: float = 0.5,
) -> dict[str, object]:
    evidence = payload["ray_evidence"]
    camera_ids = evidence["camera_ids"].numpy()
    positive = evidence["observation_type"].numpy() == 1
    positive_rows = _group_rows(camera_ids, positive)
    _, normalized_uv, _ = _owner_camera_and_uv(payload)
    changed = 0
    per_camera = []
    for camera_id, birth_rows in exact_rows.items():
        evidence_rows = positive_rows.get(camera_id)
        scale = scales.get(camera_id)
        if evidence_rows is None or scale is None:
            continue
        birth_pixels = normalized_uv[birth_rows] * np.asarray([640.0, 360.0])
        sizes = evidence["source_image_sizes"][evidence_rows].numpy()
        pixels = evidence["pixels"][evidence_rows].numpy()
        normalized = (pixels + 0.5) / np.maximum(sizes, 1)
        evidence_pixels = normalized * np.asarray([640.0, 360.0])
        distance, _ = cKDTree(birth_pixels).query(evidence_pixels, k=1)
        matched = np.isfinite(distance) & (
            distance <= float(maximum_pixel_distance)
        )
        rows = evidence_rows[matched]
        if not len(rows):
            continue
        index = torch.from_numpy(rows)
        for key in ("free_end_depth", "hit_start_depth", "hit_end_depth"):
            evidence[key][index] *= float(scale)
        changed += len(rows)
        per_camera.append(
            {
                "camera_id": int(camera_id),
                "matched_positive_rays": int(len(rows)),
                "birth_rows": int(len(birth_rows)),
                "scale": float(scale),
            }
        )
    free_end, hit_start, hit_end, interval_audit = (
        enforce_strict_foliage_ray_intervals(
            evidence["free_end_depth"].numpy(),
            evidence["hit_start_depth"].numpy(),
            evidence["hit_end_depth"].numpy(),
            evidence["observation_type"].numpy(),
        )
    )
    evidence["free_end_depth"] = torch.from_numpy(free_end)
    evidence["hit_start_depth"] = torch.from_numpy(hit_start)
    evidence["hit_end_depth"] = torch.from_numpy(hit_end)
    return {
        "changed_positive_ray_rows": int(changed),
        "exact_ray_camera_count": len(per_camera),
        "maximum_pixel_distance_at_640": float(maximum_pixel_distance),
        "interval_validation": interval_audit,
        "per_camera": per_camera,
        "contract": (
            "same_exact_pixel_gaussian_and_hit_free_interval_scaled_together"
        ),
    }


def regularize_payload(
    payload: dict,
    manifest: dict,
    images: dict[int, dict],
    cameras: dict[int, dict],
    *,
    maximum_frame_gap: int = 3,
    maximum_pixel_distance: float = 3.0,
    minimum_matches: int = 64,
    iterations: int = 2,
    maximum_canonical_distance: float = 3.0,
) -> tuple[dict, dict[str, object]]:
    by_sequence, camera_lookup, exact_rows, construction = _build_views(
        payload, manifest, images, cameras
    )
    scales, sequence_audits = _solve_all_sequences(
        by_sequence,
        maximum_frame_gap=maximum_frame_gap,
        maximum_pixel_distance=maximum_pixel_distance,
        minimum_matches=minimum_matches,
        iterations=iterations,
    )
    changed_rows, primitive_audit = _apply_primitive_scales(
        payload, scales, camera_lookup
    )
    ownership_audit = _rebind_local_ownership(
        payload,
        changed_rows,
        maximum_distance=maximum_canonical_distance,
    )
    ray_audit = _apply_exact_ray_scales(payload, scales, exact_rows)
    scale_values = np.asarray(list(scales.values()), dtype=np.float64)
    camera_rows = []
    for sequence in sorted(by_sequence):
        for view in by_sequence[sequence]:
            camera_points = (
                np.asarray(view.points, dtype=np.float64)
                @ np.asarray(view.camera.rotation, dtype=np.float64).T
                + np.asarray(view.camera.translation, dtype=np.float64)[None]
            )
            initial_median_depth = float(np.median(camera_points[:, 2]))
            scale = float(scales.get(view.camera.camera_id, 1.0))
            camera_rows.append(
                {
                    "camera_id": int(view.camera.camera_id),
                    "sequence": sequence,
                    "frame": int(view.camera.frame),
                    "scale": scale,
                    "initial_median_depth": initial_median_depth,
                    "corrected_median_depth": initial_median_depth * scale,
                    "independent_relative_rmse": float(view.relative_rmse),
                }
            )
    audit: dict[str, object] = {
        "protocol": PROTOCOL,
        "physical_problem": (
            "per_camera_affine_inverse_depth_is_ill_conditioned_in_"
            "tree_dominant_frames"
        ),
        "authority": (
            "fixed_camera_adjacent_ray_reprojection_graph_with_continuous_"
            "independent_rigid_depth_priors"
        ),
        "construction": construction,
        "sequence_graphs": sequence_audits,
        "scale_quantiles": (
            np.quantile(
                scale_values, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]
            ).tolist()
            if len(scale_values)
            else []
        ),
        "camera_scales": camera_rows,
        "primitive_update": primitive_audit,
        "local_ownership_rebind": ownership_audit,
        "ray_update": ray_audit,
        "surface_geometry_changed": False,
        "historical_trained_model_used": False,
    }
    payload["geometry_version"] = (
        str(payload.get("geometry_version", "unknown"))
        + "_sequence_metric_depth_v1"
    )
    payload.setdefault("audit", {})["sequence_metric_depth"] = audit
    return payload, audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-initialization", type=Path, required=True)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-frame-gap", type=int, default=3)
    parser.add_argument("--maximum-pixel-distance", type=float, default=3.0)
    parser.add_argument("--minimum-matches", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--maximum-canonical-distance", type=float, default=3.0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source_initialization.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
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
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        if not (output / "initialization_manifest.json").is_file():
            raise RuntimeError(
                "Refusing to replace output without initialization manifest: "
                f"{output}"
            )
        shutil.rmtree(output)

    manifest = json.loads(
        (source / "initialization_manifest.json").read_text(encoding="utf-8")
    )
    store = load_evidence_store(args.evidence_store)
    if manifest.get("evidence_hash") != store.get("evidence_hash"):
        raise RuntimeError("Initialization and Evidence Store hashes differ")
    dataset = Path(store["dataset"])
    sparse = dataset / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    payload = torch.load(
        source / "foliage_seed_gaussians.pth",
        map_location="cpu",
        weights_only=False,
    )
    payload["audit"] = {
        **manifest.get("foliage", {}),
        **payload.get("audit", {}),
    }
    payload, audit = regularize_payload(
        payload,
        manifest,
        images,
        cameras,
        maximum_frame_gap=args.maximum_frame_gap,
        maximum_pixel_distance=args.maximum_pixel_distance,
        minimum_matches=args.minimum_matches,
        iterations=args.iterations,
        maximum_canonical_distance=args.maximum_canonical_distance,
    )
    audit["producer_implementation_sha256"] = _sha256(Path(__file__))

    output.mkdir(parents=True)
    for name in ("surface_seed.npz", "surface_seed.json"):
        shutil.copy2(source / name, output / name)
    foliage_path = output / "foliage_seed_gaussians.pth"
    torch.save(payload, foliage_path)
    summary = json.loads(
        (source / "foliage_seed_gaussians.json").read_text(encoding="utf-8")
    )
    summary["sequence_metric_depth"] = audit
    summary["migrated_from"] = {
        "initialization": str(source),
        "foliage_seed_sha256": _sha256(
            source / "foliage_seed_gaussians.pth"
        ),
        "migration": PROTOCOL,
    }
    (output / "foliage_seed_gaussians.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest["version"] = PROTOCOL
    manifest["surface_seed"] = str(output / "surface_seed.npz")
    manifest["foliage_seed"] = str(foliage_path)
    manifest["foliage"] = summary
    manifest.setdefault("initialization_contract", {})[
        "sequence_metric_depth"
    ] = PROTOCOL
    manifest["migrated_from"] = {
        "initialization": str(source),
        "manifest_sha256": _sha256(source / "initialization_manifest.json"),
        "foliage_seed_sha256": _sha256(
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
                "foliage_seed_sha256": _sha256(foliage_path),
                "scale_quantiles": audit["scale_quantiles"],
                "primitive_update": audit["primitive_update"],
                "ray_update": {
                    key: value
                    for key, value in audit["ray_update"].items()
                    if key != "per_camera"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
