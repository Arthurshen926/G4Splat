"""Unified, source-aware evidence contract for outdoor reconstruction.

The store deliberately keeps COLMAP, MASt3R, Chart, plane and monocular
measurements as separate artifacts.  A fused inverse-depth cache may be
registered for efficient sampling, but it never replaces the source records.
Every persisted artifact has an immutable content hash and every sparse track
retains its source id, camera support, role posterior and uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.foliage_geometry import (
    quaternion_to_rotation,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary_with_tracks,
)
from outdoor.foliage_view_graph import sequence_id
from outdoor.scene_contract import sha256_file


EVIDENCE_STORE_VERSION = (
    "outdoor-hybrid-teacher-evidence-v5-target-raster-tracks"
)
LEGACY_EVIDENCE_STORE_VERSIONS = {
    "outdoor-hybrid-teacher-evidence-v4-crossview-consensus",
    "outdoor-hybrid-teacher-evidence-v3-contract-closed",
    "outdoor-hybrid-teacher-evidence-v2",
    "outdoor-unified-evidence-v1",
}
TRACK_EVIDENCE_VERSION = "outdoor-source-track-evidence-v1"

ROLE_RIGID = 0
ROLE_CANOPY = 1
ROLE_SKY = 2
ROLE_TRANSIENT = 3
ROLE_UNKNOWN = 4
ROLE_NAMES = ("rigid", "canopy", "sky", "transient", "unknown")


def _canonical_json_digest(payload: Any) -> str:
    value = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _copy_contract(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)


def _mask_channels(
    lookup: CambridgeMaskLookup,
    image_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = lookup.source_name_for(image_name)
    values = lookup.masks[key]
    if len(values) < 4:
        raise RuntimeError(
            f"Unified evidence requires four mask channels for {image_name}"
        )
    return tuple(
        value.detach().to(device="cpu", dtype=bool).numpy()
        for value in values[:4]
    )


def _sample_mask(
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    columns = np.clip(
        np.floor(u / float(image_width) * mask.shape[1]).astype(np.int64),
        0,
        mask.shape[1] - 1,
    )
    rows = np.clip(
        np.floor(v / float(image_height) * mask.shape[0]).astype(np.int64),
        0,
        mask.shape[0] - 1,
    )
    return mask[rows, columns]


def _project(points: np.ndarray, image: dict, camera: dict):
    rotation = quaternion_to_rotation(image["qvec"])
    camera_xyz = points @ rotation.T + image["tvec"][None]
    depth = camera_xyz[:, 2]
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = float(camera["params"][0])
        cx, cy = map(float, camera["params"][1:3])
    else:
        fx, fy, cx, cy = map(float, camera["params"][:4])
    u = camera_xyz[:, 0] / np.maximum(depth, 1e-8) * fx + cx
    v = camera_xyz[:, 1] / np.maximum(depth, 1e-8) * fy + cy
    valid = (
        (depth > 0.05)
        & np.isfinite(depth)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < camera["width"])
        & (v >= 0)
        & (v < camera["height"])
    )
    return u, v, depth, valid, max(fx, fy)


def build_track_evidence(
    sparse: Path,
    dataset: Path,
    tree_mask_pickle: Path,
    output: Path,
    *,
    source_type: str,
) -> dict[str, Any]:
    """Classify one sparse reconstruction without discarding its tracks.

    Track observations are reprojected through the fixed calibrated cameras.
    This also handles the Cambridge sparse models whose ``images.bin`` keeps
    point-track ids but intentionally omits the original 2-D rows.
    """

    sparse = Path(sparse).resolve()
    dataset = Path(dataset).resolve()
    output = Path(output)
    required = [
        sparse / "cameras.bin",
        sparse / "images.bin",
        sparse / "points3D.bin",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Sparse evidence is incomplete: " + ", ".join(map(str, missing))
        )
    cameras = read_cameras_binary(required[0])
    images = read_images_binary(required[1])
    points = read_points3d_binary_with_tracks(required[2])
    if not points:
        raise RuntimeError(f"No sparse points found in {sparse}")
    lookup = CambridgeMaskLookup(
        dataset, Path(tree_mask_pickle).resolve(), mask_indices=[0, 1, 2, 3]
    )

    xyz = np.stack([point["xyz"] for point in points]).astype(np.float32)
    rgb = np.stack([point["rgb"] for point in points]).astype(np.uint8)
    error = np.asarray([point["error"] for point in points], dtype=np.float32)
    point_ids = np.asarray([point["id"] for point in points], dtype=np.int64)
    point_index_by_id = {
        int(point_id): index for index, point_id in enumerate(point_ids)
    }
    tracks_by_image: dict[int, list[int]] = {}
    support_camera_ids: list[list[int]] = [[] for _ in points]
    for point_index, point in enumerate(points):
        for image_id in point["image_ids"]:
            image_id = int(image_id)
            if image_id in images:
                tracks_by_image.setdefault(image_id, []).append(point_index)
                support_camera_ids[point_index].append(image_id)

    role_counts = np.zeros((len(points), len(ROLE_NAMES)), dtype=np.int16)
    role_sequence_bits = np.zeros(
        (len(points), len(ROLE_NAMES)), dtype=np.uint64
    )
    depth_sum = np.zeros(len(points), dtype=np.float64)
    focal_sum = np.zeros(len(points), dtype=np.float64)
    valid_observations = np.zeros(len(points), dtype=np.int16)
    sequence_names = sorted(
        {sequence_id(image["name"]) for image in images.values()}
    )
    if len(sequence_names) > 63:
        raise RuntimeError("Track evidence supports at most 63 sequences")
    sequence_bit = {
        name: np.uint64(1 << index)
        for index, name in enumerate(sequence_names)
    }

    for image_id, point_indices_list in tracks_by_image.items():
        image = images[image_id]
        camera = cameras[image["camera_id"]]
        indices = np.asarray(point_indices_list, dtype=np.int64)
        points_world = xyz[indices].astype(np.float64)
        u, v, depth, valid, focal = _project(points_world, image, camera)
        if not np.any(valid):
            continue
        object_keep, sky_keep, distortion_keep, tree_keep = _mask_channels(
            lookup, image["name"]
        )
        object_value = _sample_mask(
            object_keep,
            u,
            v,
            image_width=camera["width"],
            image_height=camera["height"],
        )
        sky_value = _sample_mask(
            sky_keep,
            u,
            v,
            image_width=camera["width"],
            image_height=camera["height"],
        )
        distortion_value = _sample_mask(
            distortion_keep,
            u,
            v,
            image_width=camera["width"],
            image_height=camera["height"],
        )
        tree_value = _sample_mask(
            tree_keep,
            u,
            v,
            image_width=camera["width"],
            image_height=camera["height"],
        )
        transient = valid & ~object_value
        sky = valid & object_value & ~sky_value
        canopy = valid & object_value & sky_value & ~tree_value
        rigid = (
            valid
            & object_value
            & sky_value
            & tree_value
            & distortion_value
        )
        unknown = valid & ~(transient | sky | canopy | rigid)
        bit = sequence_bit[sequence_id(image["name"])]
        for role, chosen in enumerate(
            (rigid, canopy, sky, transient, unknown)
        ):
            chosen_indices = indices[chosen]
            np.add.at(role_counts[:, role], chosen_indices, 1)
            role_sequence_bits[chosen_indices, role] |= bit
        valid_indices = indices[valid]
        np.add.at(valid_observations, valid_indices, 1)
        np.add.at(depth_sum, valid_indices, depth[valid])
        np.add.at(
            focal_sum,
            valid_indices,
            np.full(valid.sum(), focal, dtype=np.float64),
        )

    role_probabilities = (
        role_counts.astype(np.float32)
        / np.maximum(valid_observations[:, None], 1)
    )
    no_observation = valid_observations == 0
    role_probabilities[no_observation] = 0.0
    role_probabilities[no_observation, ROLE_UNKNOWN] = 1.0
    dominant_role = role_probabilities.argmax(axis=1).astype(np.int8)
    sequence_counts = np.fromiter(
        (
            bin(int(np.bitwise_or.reduce(bits))).count("1")
            for bits in role_sequence_bits
        ),
        dtype=np.int16,
        count=len(points),
    )
    role_sequence_counts = np.asarray(
        [
            [bin(int(value)).count("1") for value in row]
            for row in role_sequence_bits
        ],
        dtype=np.int16,
    )
    mean_depth = depth_sum / np.maximum(valid_observations, 1)
    mean_focal = focal_sum / np.maximum(valid_observations, 1)
    # COLMAP exposes reprojection error, not a full 3-D covariance.  Convert
    # its pixel error through the observed depth/focal and explicitly record
    # this first-order model in the archive metadata.
    position_sigma = np.clip(
        (error.astype(np.float64) + 0.25)
        * mean_depth
        / np.maximum(mean_focal, 1.0)
        / np.sqrt(np.maximum(valid_observations, 1)),
        0.003,
        0.50,
    ).astype(np.float32)
    covariance_diag = np.repeat(
        position_sigma[:, None] ** 2, 3, axis=1
    ).astype(np.float32)

    offsets = np.zeros(len(points) + 1, dtype=np.int64)
    for index, values in enumerate(support_camera_ids):
        offsets[index + 1] = offsets[index] + len(values)
    flattened_camera_ids = np.asarray(
        [value for values in support_camera_ids for value in values],
        dtype=np.int32,
    )
    archive = {
        "track_id": point_ids,
        "xyz": xyz,
        "rgb": rgb,
        "reprojection_error": error,
        "valid_observation_count": valid_observations,
        "sequence_count": sequence_counts,
        "role_observation_count": role_counts,
        "role_sequence_count": role_sequence_counts,
        "role_probabilities": role_probabilities,
        "dominant_role": dominant_role,
        "position_covariance_diag": covariance_diag,
        "support_camera_offsets": offsets,
        "support_camera_ids": flattened_camera_ids,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    summary = {
        "schema_version": TRACK_EVIDENCE_VERSION,
        "source_type": str(source_type),
        "source_sparse": str(sparse),
        "point_count": len(points),
        "camera_count": len(images),
        "observed_point_count": int((valid_observations > 0).sum()),
        "dominant_role_counts": {
            name: int((dominant_role == index).sum())
            for index, name in enumerate(ROLE_NAMES)
        },
        "role_probability_order": list(ROLE_NAMES),
        "camera_support": "ragged support_camera_offsets/support_camera_ids",
        "uncertainty_model": (
            "isotropic first-order reprojection covariance: "
            "(error_px+0.25)*mean_depth/mean_focal/sqrt(observations)"
        ),
        "source_hashes": {
            path.name: sha256_file(path) for path in required
        },
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


@dataclass(frozen=True)
class EvidenceArtifact:
    name: str
    source_type: str
    path: str
    sha256: str
    bytes: int
    measurement: str
    coordinate_frame: str
    covariance: str
    camera_scope: str
    sequence_scope: str
    semantic_role: str
    validity: str

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "measurement": self.measurement,
            "coordinate_frame": self.coordinate_frame,
            "covariance": self.covariance,
            "camera_scope": self.camera_scope,
            "sequence_scope": self.sequence_scope,
            "semantic_role": self.semantic_role,
            "validity": self.validity,
        }


class EvidenceStoreBuilder:
    """Build the manifest only after all source artifacts are immutable."""

    def __init__(
        self,
        root: Path,
        *,
        dataset: Path,
        scene_contract: Path,
        semantic_contract: Path,
        split: str = "database_train",
        geometry_source: str = "mast3r_only",
        final_model: str = "hybrid_teacher",
    ):
        self.root = Path(root).resolve()
        self.dataset = Path(dataset).resolve()
        self.scene_contract = Path(scene_contract).resolve()
        self.semantic_contract = Path(semantic_contract).resolve()
        self.split = str(split)
        self.geometry_source = str(geometry_source)
        self.final_model = str(final_model)
        self.artifacts: list[EvidenceArtifact] = []

    def add_file(
        self,
        name: str,
        source_type: str,
        path: Path,
        *,
        measurement: str,
        coordinate_frame: str = "COLMAP_world",
        covariance: str = "source_specific",
        camera_scope: str = "fixed_calibrated_database",
        sequence_scope: str = "recorded_per_camera",
        semantic_role: str = "role_posterior",
        validity: str = "source_specific_mask_and_gate",
    ) -> None:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if any(artifact.name == name for artifact in self.artifacts):
            raise ValueError(f"Duplicate evidence artifact {name!r}")
        self.artifacts.append(
            EvidenceArtifact(
                name=name,
                source_type=source_type,
                path=str(path),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                measurement=measurement,
                coordinate_frame=coordinate_frame,
                covariance=covariance,
                camera_scope=camera_scope,
                sequence_scope=sequence_scope,
                semantic_role=semantic_role,
                validity=validity,
            )
        )

    def write(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        artifact_payload = [
            artifact.payload()
            for artifact in sorted(self.artifacts, key=lambda value: value.name)
        ]
        payload: dict[str, Any] = {
            "schema_version": EVIDENCE_STORE_VERSION,
            "dataset": str(self.dataset),
            "split": self.split,
            "camera_policy": "cambridge_fixed_exact_K_and_poses",
            "camera_container": (
                "cameras.bin/images.bin are serialization only; no "
                "points3D.bin or COLMAP track geometry is consumed"
            ),
            "geometry_source": self.geometry_source,
            "colmap_points_or_tracks_used": (
                self.geometry_source != "mast3r_only"
            ),
            "historical_gaussian_initialization_used": False,
            "final_model": self.final_model,
            "scene_contract": str(self.scene_contract),
            "scene_contract_sha256": sha256_file(self.scene_contract),
            "semantic_contract": str(self.semantic_contract),
            "semantic_contract_sha256": sha256_file(self.semantic_contract),
            "role_order": list(ROLE_NAMES),
            "artifacts": artifact_payload,
            "fusion_policy": {
                "source_measurements_retained": True,
                "inverse_depth_fusion_is_cache_not_ground_truth": True,
                "training_must_log_consumed_source_types": True,
            },
            "ownership_contract": {
                "rigid": "native_2d_surfel",
                "trunk_branch": "static_3d_gaussian",
                "canonical_crown": "canonical_3d_gaussian",
                "dynamic_leaf": "sequence_conditioned_3d_gaussian",
                "sky": "directional_training_field_then_standard_shell",
                "transient_unknown": "spatial_uncertainty_not_geometry",
                "canopy_surface_topology_gradient": False,
            },
            "export_contract": {
                "authoritative_model": "native_mixed_hybrid_teacher",
                "teacher_renderer_required": True,
                "standard_student": None,
                "canonical_and_conditioned_renders": True,
                "localization_assets_exclude_dynamic_leaf_and_sky": True,
            },
        }
        payload["evidence_hash"] = _canonical_json_digest(payload)
        manifest = self.root / "evidence_manifest.json"
        manifest.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return payload


def load_evidence_store(path: Path, *, verify_hashes: bool = True) -> dict:
    path = Path(path)
    manifest = path / "evidence_manifest.json" if path.is_dir() else path
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        EVIDENCE_STORE_VERSION,
        *LEGACY_EVIDENCE_STORE_VERSIONS,
    }:
        raise RuntimeError(f"Unsupported evidence store: {manifest}")
    evidence_hash = payload.get("evidence_hash")
    unhashed = dict(payload)
    unhashed.pop("evidence_hash", None)
    if evidence_hash != _canonical_json_digest(unhashed):
        raise RuntimeError("Evidence manifest hash does not match its content")
    if verify_hashes:
        for artifact in payload["artifacts"]:
            artifact_path = Path(artifact["path"])
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise RuntimeError(
                    f"Evidence artifact changed after registration: {artifact_path}"
                )
    return payload


def artifact_path(
    store: dict,
    name: str,
    *,
    required: bool = True,
) -> Path | None:
    for artifact in store["artifacts"]:
        if artifact["name"] == name:
            return Path(artifact["path"])
    if required:
        raise KeyError(f"Evidence store has no artifact {name!r}")
    return None
