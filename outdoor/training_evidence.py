"""Lazy per-view geometry sources consumed by the unified teacher."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from outdoor.evidence_store import artifact_path, load_evidence_store
from outdoor.inverse_depth import INVERSE_DEPTH_FUSION_VERSION


class OutdoorGeometryEvidence:
    """Keep every geometry source separate at loss construction time."""

    def __init__(self, evidence_store: Path):
        self.store = load_evidence_store(evidence_store, verify_hashes=False)
        chart_path = artifact_path(
            self.store, "chart_geometry", required=False
        )
        chart_camera_path = artifact_path(
            self.store, "chart_cameras", required=False
        )
        self.chart = None
        self.chart_scale_factor = 1.0
        self.frame_by_stem: dict[str, int] = {}
        if chart_path is not None and chart_camera_path is not None:
            archive = np.load(chart_path, allow_pickle=False)
            self.chart_scale_factor = float(archive["scale_factor"])
            if (
                not np.isfinite(self.chart_scale_factor)
                or self.chart_scale_factor <= 0
            ):
                raise RuntimeError(
                    "MAtCha chart scale_factor must be finite and positive"
                )
            # Materialize only the fields used by training.  Keeping the
            # pointmaps out saves hundreds of MiB in every worker.
            self.chart = {
                # All renderers and fixed Cambridge cameras use the original
                # world scale.  MAtCha stores atlas depths in its normalized
                # optimization scale.
                "depth": (
                    archive["depths"].astype(np.float32)
                    / self.chart_scale_factor
                ),
                "confidence": archive["confs"].astype(np.float32),
                "reference_mask": archive.get(
                    "alignment_reference_mask",
                    np.ones(archive["depths"].shape, dtype=bool),
                ).astype(bool),
                "active": (
                    archive.get(
                        "quality_selection_active",
                        np.ones(archive["depths"].shape[0], dtype=bool),
                    ).astype(bool)
                    & archive.get(
                        "alignment_gate_valid",
                        np.ones(archive["depths"].shape[0], dtype=bool),
                    ).astype(bool)
                ),
            }
            archive.close()
            cameras = json.loads(
                chart_camera_path.read_text(encoding="utf-8")
            )
            for index, value in enumerate(cameras["filepaths"]):
                self.frame_by_stem[Path(value).stem] = index
        plane_marker = artifact_path(
            self.store, "plane_depth_root_marker", required=False
        )
        inverse_marker = artifact_path(
            self.store, "inverse_depth_root_marker", required=False
        )
        self.plane_root = (
            plane_marker.parent if plane_marker is not None else None
        )
        self.inverse_root = (
            inverse_marker.parent if inverse_marker is not None else None
        )
        self.inverse_manifest = None
        inverse_manifest_path = artifact_path(
            self.store, "inverse_depth_manifest", required=False
        )
        if inverse_manifest_path is not None:
            self.inverse_manifest = json.loads(
                inverse_manifest_path.read_text(encoding="utf-8")
            )
            version = self.inverse_manifest.get("schema_version")
            if version != INVERSE_DEPTH_FUSION_VERSION:
                raise RuntimeError(
                    "Refusing a mixed/unknown-unit inverse-depth cache: "
                    f"{version!r}; expected {INVERSE_DEPTH_FUSION_VERSION!r}. "
                    "Rebuild the hybrid Teacher evidence store."
                )
        self._verify_array_index("plane_array_index")
        self._verify_array_index("inverse_depth_array_index")
        dav2_index_path = artifact_path(
            self.store, "dav2_index", required=False
        )
        self.dav2_records: dict[str, Path] = {}
        if dav2_index_path is not None:
            payload = json.loads(
                dav2_index_path.read_text(encoding="utf-8")
            )
            self.dav2_records = {
                stem: Path(record["path"])
                for stem, record in payload.get("records", {}).items()
            }
        self.track_archives = {}
        for source_code, artifact_name in (
            (0, "colmap_tracks"),
            (1, "mast3r_tracks"),
        ):
            path = artifact_path(
                self.store, artifact_name, required=False
            )
            if path is None:
                continue
            with np.load(path, allow_pickle=False) as archive:
                order = np.argsort(archive["track_id"])
                self.track_archives[source_code] = {
                    "track_id": archive["track_id"][order].astype(np.int64),
                    "xyz": archive["xyz"][order].astype(np.float32),
                    "variance": archive["position_covariance_diag"][
                        order
                    ].astype(np.float32),
                }
        structure_path = artifact_path(
            self.store, "structure_graph", required=False
        )
        self.structure = None
        if structure_path is not None:
            with np.load(structure_path, allow_pickle=False) as archive:
                self.structure = {
                    "centers": (
                        archive["unit_centers"].astype(np.float32)
                        / self.chart_scale_factor
                    ),
                    "weight": archive["unit_weight"].astype(np.float32),
                    "view_count": archive["unit_view_count"].astype(np.float32),
                    "block_id": archive["unit_block_ids"].astype(np.int32),
                }
        self._track_device_cache = {}
        self._structure_device_cache = {}
        self.consumed = {
            "chart": 0,
            "plane": 0,
            "inverse_depth": 0,
            "dav2": 0,
            "track_factor": 0,
            "structure_factor": 0,
        }
        self.preflight = self._metric_preflight()

    def _verify_array_index(self, artifact_name: str) -> None:
        index_path = artifact_path(
            self.store, artifact_name, required=False
        )
        if index_path is None:
            raise RuntimeError(
                f"Evidence store is missing {artifact_name}; rebuild it"
            )
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            raise RuntimeError(f"{artifact_name} has no records")
        for record in records:
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
            ):
                raise RuntimeError(
                    f"Geometry array changed or disappeared: {path}"
                )
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise RuntimeError(
                    f"Geometry array hash mismatch: {path}"
                )

    def _metric_preflight(self) -> dict:
        """Prove source units and non-empty owner support before optimization."""
        if self.chart is None or self.inverse_root is None:
            raise RuntimeError(
                "Hybrid Teacher requires Chart and world-metric inverse depth"
            )
        chart_owner_pixels = 0
        chart_effective_pixels = 0
        plane_owner_pixels = 0
        plane_chart_ratios: list[float] = []
        frame_count = int(self.chart["depth"].shape[0])
        for frame in range(frame_count):
            inverse = self.inverse_root
            source_path = inverse / f"source_bitmask_frame{frame:06d}.npy"
            support_path = inverse / f"support_view_count_frame{frame:06d}.npy"
            rho_path = inverse / f"rho_mean_frame{frame:06d}.npy"
            variance_path = inverse / f"rho_variance_frame{frame:06d}.npy"
            if not all(
                path.is_file()
                for path in (
                    source_path,
                    support_path,
                    rho_path,
                    variance_path,
                )
            ):
                raise FileNotFoundError(
                    f"Incomplete inverse-depth frame {frame:06d}"
                )
            source = np.load(source_path).astype(np.uint8, copy=False)
            support = np.load(support_path)
            rho = np.load(rho_path)
            variance = np.load(variance_path)
            plane_owner = (source & 1) != 0
            chart_owner = ((source & 2) != 0) & ~plane_owner
            valid_inverse = (
                np.isfinite(rho)
                & (rho > 0)
                & np.isfinite(variance)
                & (variance > 0)
            )
            chart_owner_pixels += int(chart_owner.sum())
            chart_effective_pixels += int(
                (chart_owner & (support >= 1) & valid_inverse).sum()
            )
            plane_owner_pixels += int(plane_owner.sum())

            plane_path = (
                self.plane_root
                / f"plane_depth_frame{frame:06d}.npy"
                if self.plane_root is not None
                else None
            )
            if (
                plane_path is not None
                and plane_path.is_file()
                and bool(plane_owner.any())
                and self.chart["active"][frame]
            ):
                plane = np.load(plane_path).astype(np.float32, copy=False)
                chart = self.chart["depth"][frame]
                chart = np.asarray(
                    Image.fromarray(chart, mode="F").resize(
                        (source.shape[1], source.shape[0]),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
                valid = (
                    plane_owner
                    & np.isfinite(plane)
                    & (plane > 0)
                    & np.isfinite(chart)
                    & (chart > 0)
                )
                if bool(valid.any()):
                    plane_chart_ratios.append(
                        float(np.median(plane[valid] / chart[valid]))
                    )
        if chart_owner_pixels <= 0 or chart_effective_pixels <= 0:
            raise RuntimeError(
                "Chart inverse-depth owner has zero effective pixels; "
                "source support/arbitration is invalid"
            )
        ratio = (
            float(np.median(plane_chart_ratios))
            if plane_chart_ratios
            else None
        )
        if ratio is not None and not 0.5 <= ratio <= 2.0:
            raise RuntimeError(
                "Plane/Chart metric depth scale mismatch: median ratio "
                f"{ratio:.6g}; expected the same Cambridge camera scale"
            )
        return {
            "frame_count": frame_count,
            "plane_owner_pixels": plane_owner_pixels,
            "chart_owner_pixels": chart_owner_pixels,
            "chart_effective_inverse_pixels": chart_effective_pixels,
            "chart_effective_fraction": (
                chart_effective_pixels / chart_owner_pixels
            ),
            "plane_chart_depth_ratio_median": ratio,
        }

    def track_factor(
        self, surface, *, maximum_tracks: int = 8192
    ) -> tuple[torch.Tensor, dict]:
        """Robust persistent SfM/MASt3R 3D track factor for seed primitives."""
        zero = surface.get_xyz.new_zeros(())
        if not self.track_archives or not hasattr(surface, "_track_id"):
            return zero, {"matched": 0}
        device = surface.get_xyz.device
        device_key = str(device)
        if device_key not in self._track_device_cache:
            self._track_device_cache[device_key] = {
                source: {
                    name: torch.from_numpy(value).to(device=device)
                    for name, value in archive.items()
                }
                for source, archive in self.track_archives.items()
            }
        total = zero
        total_weight = zero
        matched_total = 0
        for source, archive in self._track_device_cache[device_key].items():
            candidate = (
                (surface._source_type == int(source))
                & (surface._track_id >= 0)
            )
            indices = torch.nonzero(candidate, as_tuple=False).flatten()
            if not len(indices):
                continue
            if len(indices) > maximum_tracks:
                stride = int(np.ceil(len(indices) / maximum_tracks))
                indices = indices[::stride][:maximum_tracks]
            ids = surface._track_id[indices]
            position = torch.searchsorted(archive["track_id"], ids)
            in_range = position < len(archive["track_id"])
            safe_position = position.clamp_max(
                max(len(archive["track_id"]) - 1, 0)
            )
            matched = in_range & (
                archive["track_id"][safe_position] == ids
            )
            if not bool(matched.any()):
                continue
            indices = indices[matched]
            position = safe_position[matched]
            variance = archive["variance"][position].clamp_min(1e-6)
            delta = surface.get_xyz[indices] - archive["xyz"][position]
            mahalanobis = (delta.square() / variance).sum(-1)
            weight = surface._geometry_confidence[indices].clamp(0.05, 1)
            total = total + (torch.log1p(mahalanobis) * weight).sum()
            total_weight = total_weight + weight.sum()
            matched_total += int(len(indices))
        if matched_total:
            self.consumed["track_factor"] += 1
        return total / total_weight.clamp_min(1), {
            "matched": matched_total,
        }

    def structure_factor(
        self, surface, *, maximum_points: int = 2048
    ) -> tuple[torch.Tensor, dict]:
        """Persistent MAtCha/G4 structural-unit support for rigid surfels."""
        zero = surface.get_xyz.new_zeros(())
        if self.structure is None or not len(surface.get_xyz):
            return zero, {"matched": 0, "blocks": 0}
        device = surface.get_xyz.device
        key = str(device)
        if key not in self._structure_device_cache:
            self._structure_device_cache[key] = {
                name: torch.from_numpy(value).to(device=device)
                for name, value in self.structure.items()
            }
        structure = self._structure_device_cache[key]
        # Structural units come from the MAtCha/Chart atlas and own only its
        # residual seeds (source_type=2). MASt3R track anchors already have a
        # covariance-aware track factor; applying both factors to them is an
        # ownership conflict.
        owned = torch.nonzero(
            surface._source_type == 2, as_tuple=False
        ).flatten()
        if not len(owned):
            return zero, {"matched": 0, "blocks": 0}
        if len(owned) > maximum_points:
            stride = int(np.ceil(len(owned) / maximum_points))
            owned = owned[::stride][:maximum_points]
        points = surface.get_xyz[owned]
        # 2k x 2.2k is bounded and keeps the selected unit differentiable
        # with respect to the surfel position.
        distance = torch.cdist(points, structure["centers"])
        nearest_distance, nearest = distance.min(dim=1)
        confidence = (
            structure["weight"][nearest].clamp_min(0.05)
            * structure["view_count"][nearest].clamp_min(1).sqrt()
        )
        # Units summarize finite facade patches, not exact point targets.
        # A 4 cm dead zone preserves texture-driven substructure.
        residual = (nearest_distance - 0.04).clamp_min(0)
        loss = (
            torch.log1p((residual / 0.08).square()) * confidence
        ).sum() / confidence.sum().clamp_min(1)
        self.consumed["structure_factor"] += 1
        return loss, {
            "matched": int(len(points)),
            "blocks": int(
                structure["block_id"][nearest].unique().numel()
            ),
        }

    @property
    def geometry_view_stems(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.frame_by_stem) | set(self.dav2_records))
        )

    @property
    def metric_view_stems(self) -> tuple[str, ...]:
        return tuple(sorted(self.frame_by_stem))

    @property
    def ordinal_view_stems(self) -> tuple[str, ...]:
        return tuple(sorted(self.dav2_records))

    @staticmethod
    def _load_monocular_depth(path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            value = np.load(path)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for key in ("depth", "pred", "prediction", "arr_0"):
                    if key in archive:
                        value = archive[key]
                        break
                else:
                    raise KeyError(f"No depth array in {path}")
        else:
            value = np.asarray(Image.open(path))
        value = np.asarray(value, dtype=np.float32)
        value = np.squeeze(value)
        if value.ndim != 2:
            raise ValueError(f"DAV2 depth must be HxW, got {value.shape}")
        return value

    @staticmethod
    def _tensor(
        value: np.ndarray,
        *,
        device: torch.device,
        shape: tuple[int, int],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        tensor = torch.from_numpy(
            np.array(value, copy=True, order="C")
        ).to(device=device, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)[None]
        else:
            raise ValueError(f"Unsupported evidence shape {value.shape}")
        if tensor.shape[-2:] != shape:
            tensor = F.interpolate(
                tensor,
                size=shape,
                mode=mode,
                align_corners=False if mode != "nearest" else None,
            )
        return tensor[0]

    def fields(
        self,
        image_name: str,
        *,
        device: torch.device,
        shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        stem = Path(str(image_name)).stem
        frame = self.frame_by_stem.get(stem)
        result: dict[str, torch.Tensor] = {}
        if (
            frame is not None
            and self.chart is not None
            and self.chart["active"][frame]
        ):
            depth = self.chart["depth"][frame]
            confidence = self.chart["confidence"][frame]
            valid = (
                self.chart["reference_mask"][frame]
                & np.isfinite(depth)
                & (depth > 0)
                & np.isfinite(confidence)
                & (confidence > 0)
            )
            if np.any(valid):
                median_depth = float(np.median(depth[valid]))
                absolute_limit = max(6.0 * median_depth, 5.0)
                valid &= depth <= absolute_limit
            confidence = confidence / max(
                float(np.median(confidence[valid]))
                if np.any(valid)
                else 1.0,
                1e-6,
            )
            result["chart_depth"] = self._tensor(
                depth, device=device, shape=shape
            )
            result["chart_weight"] = self._tensor(
                (valid * np.clip(confidence, 0, 4)).astype(np.float32),
                device=device,
                shape=shape,
            )
            self.consumed["chart"] += 1
        if self.plane_root is not None and frame is not None:
            plane_depth = self.plane_root / f"plane_depth_frame{frame:06d}.npy"
            plane_confidence = (
                self.plane_root
                / f"plane_confidence_frame{frame:06d}.npy"
            )
            plane_normal = (
                self.plane_root
                / f"depth_normal_world_frame{frame:06d}.npy"
            )
            if plane_depth.is_file() and plane_confidence.is_file():
                depth = np.load(plane_depth)
                confidence = np.load(plane_confidence)
                valid = (
                    np.isfinite(depth)
                    & (depth > 0)
                    & np.isfinite(confidence)
                    & (confidence > 0)
                )
                result["plane_depth"] = self._tensor(
                    depth, device=device, shape=shape
                )
                result["plane_weight"] = self._tensor(
                    (valid * np.clip(confidence, 0, 1)).astype(
                        np.float32
                    ),
                    device=device,
                    shape=shape,
                )
                if plane_normal.is_file():
                    result["plane_normal_world"] = self._tensor(
                        np.load(plane_normal),
                        device=device,
                        shape=shape,
                    )
                self.consumed["plane"] += 1
            mono_depth = (
                self.plane_root / f"mono_depth_frame{frame:06d}.tiff"
            )
            if mono_depth.is_file():
                result["mono_depth"] = self._tensor(
                    np.asarray(Image.open(mono_depth), dtype=np.float32),
                    device=device,
                    shape=shape,
                )
        if self.inverse_root is not None and frame is not None:
            paths = {
                "rho_mean": self.inverse_root
                / f"rho_mean_frame{frame:06d}.npy",
                "rho_variance": self.inverse_root
                / f"rho_variance_frame{frame:06d}.npy",
                "source_bitmask": self.inverse_root
                / f"source_bitmask_frame{frame:06d}.npy",
                "support_view_count": self.inverse_root
                / f"support_view_count_frame{frame:06d}.npy",
            }
            if all(path.is_file() for path in paths.values()):
                for name, path in paths.items():
                    mode = (
                        "nearest"
                        if name in {"source_bitmask", "support_view_count"}
                        else "bilinear"
                    )
                    value = np.load(path)
                    result[name] = self._tensor(
                        value,
                        device=device,
                        shape=shape,
                        mode=mode,
                    )
                self.consumed["inverse_depth"] += 1
        dav2_path = self.dav2_records.get(stem)
        if dav2_path is not None and dav2_path.is_file():
            result["mono_depth"] = self._tensor(
                self._load_monocular_depth(dav2_path),
                device=device,
                shape=shape,
            )
            self.consumed["dav2"] += 1
        return result

    def audit(self) -> dict:
        return {
            "available_geometry_views": len(self.frame_by_stem),
            "available_dav2_views": len(self.dav2_records),
            "source_consumption_count": dict(self.consumed),
            "source_losses_are_mutually_exclusive": True,
            "inverse_depth_is_cache_not_replacement": True,
            "chart_to_cambridge_world_scale": float(
                1.0 / self.chart_scale_factor
            ),
            "metric_depth_coordinate_frame": "cambridge_fixed_camera_depth",
            "inverse_depth_schema": (
                None
                if self.inverse_manifest is None
                else self.inverse_manifest.get("schema_version")
            ),
            "metric_preflight": dict(self.preflight),
            "all_metric_depth_factors_use_cambridge_world_scale": True,
        }
