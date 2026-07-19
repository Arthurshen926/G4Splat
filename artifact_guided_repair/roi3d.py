from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .core import CameraGeometry, project_points, rasterize_projected_component


def _json_ready(value: object) -> object:
    """Convert evidence metadata without discarding its audit structure."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class ArtifactROI3D:
    """A bounded, evidence-carrying 3D repair region.

    The point support is the authoritative cross-view representation. A plane
    is optional and is used only after robust clean-view validation.
    """

    roi_id: str
    points: np.ndarray
    source_view_ids: tuple[int, ...]
    classification: str
    plane_world: np.ndarray | None = None
    slab_thickness: float = 0.0
    evidence_view_ids: tuple[int, ...] = ()
    chart_ids: tuple[int, ...] = ()
    surfel_ids: tuple[int, ...] = ()
    gaussian_ids: tuple[int, ...] = ()
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        if not len(self.points):
            raise ValueError("ArtifactROI3D requires at least one 3D point")
        finite = np.isfinite(self.points).all(axis=1)
        if not finite.all():
            raise ValueError("ArtifactROI3D points must be finite")
        if self.plane_world is not None:
            self.plane_world = np.asarray(self.plane_world, dtype=np.float64).reshape(4)
            normal_norm = float(np.linalg.norm(self.plane_world[:3]))
            if normal_norm <= 1e-12:
                raise ValueError("plane_world has a zero normal")
            self.plane_world = self.plane_world / normal_norm
        if self.slab_thickness < 0:
            raise ValueError("slab_thickness must be non-negative")

    @property
    def center(self) -> np.ndarray:
        return np.median(self.points, axis=0)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def project_mask(
        self,
        camera: CameraGeometry,
        *,
        reference_depth: np.ndarray | None = None,
        relative_depth_tolerance: float = 0.10,
        point_radius: int = 5,
        close_radius: int = 7,
        max_area_fraction: float = 0.50,
    ) -> np.ndarray:
        points = self.points
        if reference_depth is not None:
            depth = np.asarray(reference_depth, dtype=np.float32).squeeze()
            if depth.shape != (camera.height, camera.width):
                raise ValueError(
                    f"reference_depth shape {depth.shape} does not match camera "
                    f"{(camera.height, camera.width)}"
                )
            pixels, z, inside = project_points(points, camera)
            rounded = np.rint(pixels).astype(np.int64)
            indices = np.flatnonzero(inside)
            if len(indices):
                x = rounded[indices, 0]
                y = rounded[indices, 1]
                observed = depth[y, x]
                valid = np.isfinite(observed) & (observed > 0)
                consistent = np.zeros(len(indices), dtype=bool)
                consistent[valid] = (
                    np.abs(z[indices][valid] - observed[valid])
                    / np.maximum(observed[valid], 1e-6)
                    <= relative_depth_tolerance
                )
                points = points[indices[consistent]]
            else:
                points = points[:0]
        if not len(points):
            return np.zeros((camera.height, camera.width), dtype=bool)
        return rasterize_projected_component(
            points,
            camera,
            point_radius=point_radius,
            close_radius=close_radius,
            max_area_fraction=max_area_fraction,
        )

    def summary(self) -> dict[str, object]:
        lower, upper = self.bounds
        return {
            "roi_id": self.roi_id,
            "classification": self.classification,
            "point_count": int(len(self.points)),
            "source_view_ids": list(self.source_view_ids),
            "evidence_view_ids": list(self.evidence_view_ids),
            "center_world": self.center.tolist(),
            "bounds_world": [lower.tolist(), upper.tolist()],
            "plane_world": None if self.plane_world is None else self.plane_world.tolist(),
            "slab_thickness": float(self.slab_thickness),
            "chart_ids": list(self.chart_ids),
            "surfel_ids": list(self.surfel_ids),
            "gaussian_ids": list(self.gaussian_ids),
            "evidence": _json_ready(self.evidence),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points.astype(np.float32),
            plane_world=(
                np.empty((0,), dtype=np.float32)
                if self.plane_world is None
                else self.plane_world.astype(np.float32)
            ),
            metadata=np.asarray(json.dumps(self.summary())),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ArtifactROI3D":
        with np.load(path) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            plane = np.asarray(payload["plane_world"])
            return cls(
                roi_id=metadata["roi_id"],
                points=np.asarray(payload["points"]),
                source_view_ids=tuple(metadata["source_view_ids"]),
                classification=metadata["classification"],
                plane_world=None if not plane.size else plane,
                slab_thickness=float(metadata["slab_thickness"]),
                evidence_view_ids=tuple(metadata["evidence_view_ids"]),
                chart_ids=tuple(metadata["chart_ids"]),
                surfel_ids=tuple(metadata["surfel_ids"]),
                gaussian_ids=tuple(metadata["gaussian_ids"]),
                evidence=dict(metadata.get("evidence", {})),
            )
