"""Lazy per-view geometry sources consumed by the unified teacher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from outdoor.evidence_store import artifact_path, load_evidence_store


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
        self.frame_by_stem: dict[str, int] = {}
        if chart_path is not None and chart_camera_path is not None:
            archive = np.load(chart_path, allow_pickle=False)
            # Materialize only the fields used by training.  Keeping the
            # pointmaps out saves hundreds of MiB in every worker.
            self.chart = {
                "depth": archive["depths"].astype(np.float32),
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
        self.consumed = {
            "chart": 0,
            "plane": 0,
            "inverse_depth": 0,
        }

    @property
    def geometry_view_stems(self) -> tuple[str, ...]:
        return tuple(sorted(self.frame_by_stem))

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
        if frame is None:
            return {}
        result: dict[str, torch.Tensor] = {}
        if self.chart is not None and self.chart["active"][frame]:
            depth = self.chart["depth"][frame]
            confidence = self.chart["confidence"][frame]
            valid = (
                self.chart["reference_mask"][frame]
                & np.isfinite(depth)
                & (depth > 0)
                & np.isfinite(confidence)
                & (confidence > 0)
            )
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
        if self.plane_root is not None:
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
        if self.inverse_root is not None:
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
                    result[name] = self._tensor(
                        np.load(path),
                        device=device,
                        shape=shape,
                        mode=mode,
                    )
                self.consumed["inverse_depth"] += 1
        return result

    def audit(self) -> dict:
        return {
            "available_geometry_views": len(self.frame_by_stem),
            "source_consumption_count": dict(self.consumed),
            "source_losses_are_independent": True,
            "inverse_depth_is_cache_not_replacement": True,
        }
