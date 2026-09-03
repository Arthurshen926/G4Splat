"""Permanent Chart-surface evidence independent of render topology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from outdoor.moge3_chart_base import (
    MOGE3_CHART_BASE_SOURCES,
    load_chart_base,
)
from outdoor.training_evidence import EvidenceEpochSampler


class ChartSurfaceModel:
    """Immutable Chart evidence plus a short-lived renderer-seed prior.

    The permanent source-resolution Chart depth factor is evaluated by
    ``OutdoorGeometryEvidence.chart_native_factor`` and never depends on a
    Gaussian identity.  This class retains UV/adjacency metadata and supplies
    only the bootstrap centre/Jacobian prior before renderer witnesses are
    released for normal local replacement.
    """

    def __init__(
        self,
        surface_seed: Path,
        *,
        evidence_store: Path | None = None,
        chart_base_source: str = "matcha",
        seed: int = 3253,
        maximum_tangent_scale: float = float("inf"),
    ):
        if maximum_tangent_scale <= 0:
            raise ValueError("maximum_tangent_scale must be positive")
        if chart_base_source not in MOGE3_CHART_BASE_SOURCES:
            raise ValueError(
                f"Unsupported Chart base source: {chart_base_source!r}"
            )
        self.chart_base_source = str(chart_base_source)
        with np.load(surface_seed, allow_pickle=False) as archive:
            source = archive["source_type"].astype(np.int8)
            all_selected = source == 2
            all_xyz = archive["xyz"][all_selected].astype(np.float32)
            all_scales = archive["scales"][all_selected].astype(np.float32)
            all_evidence_id = archive["track_id"][all_selected].astype(
                np.int64
            )
            all_chart_id = archive["chart_id"][all_selected].astype(np.int32)
            all_uv = archive["chart_uv"][all_selected].astype(np.float32)
            selected = source == 2
            if "persistent_geometry_evidence" in archive:
                selected &= archive[
                    "persistent_geometry_evidence"
                ].astype(bool)
            self.xyz = archive["xyz"][selected].astype(np.float32)
            self.sigma = archive["position_sigma"][selected].astype(np.float32)
            self.footprint = archive["scales"][selected].max(axis=1).astype(
                np.float32
            )
            self.evidence_id = archive["track_id"][selected].astype(np.int64)
            self.chart_id = archive["chart_id"][selected].astype(np.int32)
            self.uv = archive["chart_uv"][selected].astype(np.float32)
            self.coverage_only_seed_count = int(
                (
                    (source == 2)
                    & ~archive.get(
                        "persistent_geometry_evidence",
                        np.ones(len(source), dtype=bool),
                    ).astype(bool)
                ).sum()
            )
        atlas_payload = None
        if evidence_store is not None:
            atlas_payload = LearnableInverseDepthAtlas._load_payload(
                Path(evidence_store),
                chart_base_source=self.chart_base_source,
            )
        if atlas_payload is not None and self.chart_base_source != "matcha":
            all_xyz = LearnableInverseDepthAtlas._seed_xyz_from_payload(
                atlas_payload,
                all_chart_id,
                all_uv,
                all_xyz,
            )
            self.xyz = LearnableInverseDepthAtlas._seed_xyz_from_payload(
                atlas_payload,
                self.chart_id,
                self.uv,
                self.xyz,
            )
        if len(self.xyz) and (
            np.any(self.evidence_id >= -1)
            or len(np.unique(self.evidence_id)) != len(self.evidence_id)
        ):
            raise RuntimeError("Chart evidence ids must be unique and below -1")
        if len(self.xyz) and not np.isfinite(self.uv).all():
            raise RuntimeError("Chart-bound anchors require finite UV coordinates")
        self.sampler = EvidenceEpochSampler(len(self.xyz), seed=seed)
        edges = []
        for chart in np.unique(self.chart_id):
            rows = np.flatnonzero(self.chart_id == chart)
            if len(rows) < 2:
                continue
            neighbours = cKDTree(self.uv[rows]).query(
                self.uv[rows], k=min(5, len(rows))
            )[1]
            for local, adjacent in enumerate(np.atleast_2d(neighbours)):
                for other in np.atleast_1d(adjacent)[1:]:
                    a, b = int(rows[local]), int(rows[int(other)])
                    if a != b:
                        edges.append((min(a, b), max(a, b)))
        self.edges = (
            np.unique(np.asarray(edges, dtype=np.int64), axis=0)
            if edges
            else np.empty((0, 2), dtype=np.int64)
        )
        self.edge_sampler = EvidenceEpochSampler(
            len(self.edges), seed=seed + 1
        )
        # ``persistent_geometry_evidence`` is a subset of the renderer seed
        # rows. Its negative evidence ids consequently contain gaps whenever
        # an unsupported coverage-only Chart sample lies between two
        # confirmed samples. Never decode an id as ``-(row + 2)`` after this
        # filtering: that aliases a confirmed observation to the wrong atlas
        # row. Keep an explicit sorted id -> compact evidence-row lookup.
        self.evidence_id_order = np.argsort(self.evidence_id)
        self.sorted_evidence_id = self.evidence_id[
            self.evidence_id_order
        ].copy()
        self._cache = {}
        self.factor_calls = 0
        self.anchor_rows_sampled = 0
        self.anchor_rows_matched = 0
        self.edge_rows_sampled = 0
        self.edge_rows_matched = 0
        self.factor_calls_with_matched_edges = 0
        self.atlas = (
            LearnableInverseDepthAtlas(
                evidence_store=Path(evidence_store),
                seed_xyz=all_xyz,
                evidence_id=all_evidence_id,
                chart_id=all_chart_id,
                uv=all_uv,
                tangent_scale=all_scales,
                maximum_tangent_scale=maximum_tangent_scale,
                chart_base_source=self.chart_base_source,
                payload=atlas_payload,
            )
            if evidence_store is not None
            else None
        )

    def parameters(self):
        """Expose only the training-time atlas parameters."""
        if self.atlas is None:
            return iter(())
        return self.atlas.parameters()

    def cuda(self):
        if self.atlas is not None:
            self.atlas.cuda()
        return self

    def geometry_override(self, surface):
        if self.atlas is None:
            return None
        return self.atlas.geometry_override(surface)

    def live_surface_rows(self, surface) -> torch.Tensor:
        """Return the surface rows whose geometry is owned by the atlas.

        Screen-space footprint feedback is produced by the rasterizer in
        surface-row order.  Exposing that ownership explicitly lets the
        trainer persist a measured footprint ceiling on live Chart rows even
        under an ``atlas_residual`` handoff; otherwise only completion rows
        were corrected and the immutable Chart prefix could grow back into
        large paint splats through the atlas override.
        """
        if self.atlas is None:
            return torch.empty(
                0, dtype=torch.long, device=surface.get_xyz.device
            )
        return self.atlas.live_surface_rows(surface)

    def regularizer(self) -> torch.Tensor:
        if self.atlas is None:
            if self._cache:
                return next(iter(self._cache.values()))["xyz"].new_zeros(())
            return torch.zeros(())
        return self.atlas.regularizer()

    def capture(self) -> dict:
        return {
            "version": "chart-surface-model-v2",
            "sampler": self.sampler.capture_state(),
            "edge_sampler": self.edge_sampler.capture_state(),
            "runtime": {
                "factor_calls": self.factor_calls,
                "anchor_rows_sampled": self.anchor_rows_sampled,
                "anchor_rows_matched": self.anchor_rows_matched,
                "edge_rows_sampled": self.edge_rows_sampled,
                "edge_rows_matched": self.edge_rows_matched,
                "factor_calls_with_matched_edges": (
                    self.factor_calls_with_matched_edges
                ),
            },
            "atlas": None if self.atlas is None else self.atlas.capture(),
        }

    def restore(self, state: dict | None) -> None:
        if state is None:
            return
        if state.get("version") != "chart-surface-model-v2":
            raise RuntimeError("Unsupported Chart surface checkpoint")
        self.sampler.restore_state(state["sampler"])
        self.edge_sampler.restore_state(state["edge_sampler"])
        for name, value in state.get("runtime", {}).items():
            if hasattr(self, name):
                setattr(self, name, int(value))
        if self.atlas is not None and state.get("atlas") is not None:
            self.atlas.restore(state["atlas"])

    @torch.no_grad()
    def adapt_uv_quadtree(
        self,
        surface,
        *,
        maximum_net_growth: int,
        maximum_points: int,
        gradient_threshold: float,
        radius_target: float = 8.0,
        maximum_level: int = 3,
        priority_snapshot: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        if self.atlas is None:
            return {
                "enabled": False,
                "parents_replaced": 0,
                "children": 0,
                "net_growth": 0,
            }
        return self.atlas.adapt_uv_quadtree(
            surface,
            maximum_net_growth=maximum_net_growth,
            maximum_points=maximum_points,
            gradient_threshold=gradient_threshold,
            radius_target=radius_target,
            maximum_level=maximum_level,
            priority_snapshot=priority_snapshot,
        )

    @torch.no_grad()
    def topology_priority_snapshot(
        self,
        surface,
        *,
        gradient_threshold: float,
        radius_target: float = 8.0,
    ) -> dict[str, torch.Tensor] | None:
        if self.atlas is None:
            return None
        return self.atlas.topology_priority_snapshot(
            surface,
            gradient_threshold=gradient_threshold,
            radius_target=radius_target,
        )

    @torch.no_grad()
    def bake_into_surface(self, surface) -> dict:
        if self.atlas is None:
            return {"baked_rows": 0}
        return self.atlas.bake_into_surface(surface)

    def _device(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = {
                "xyz": torch.from_numpy(self.xyz).to(device),
                "sigma": torch.from_numpy(self.sigma).to(device),
                "footprint": torch.from_numpy(self.footprint).to(device),
                "evidence_id": torch.from_numpy(self.evidence_id).to(device),
                "sorted_evidence_id": torch.from_numpy(
                    self.sorted_evidence_id
                ).to(device),
                "evidence_id_order": torch.from_numpy(
                    self.evidence_id_order
                ).to(device),
                "edges": torch.from_numpy(self.edges).to(device),
            }
        return self._cache[key]

    def factor(self, surface, *, maximum_anchors: int = 4096):
        self.factor_calls += 1
        zero = surface.get_xyz.new_zeros(())
        if not len(self.xyz):
            return zero, {
                "matched": 0,
                "live_chart_rows": 0,
                "uv_edges_sampled": 0,
                "uv_edges_matched": 0,
                "coverage": self.sampler.audit(),
                "edge_coverage": self.edge_sampler.audit(),
            }
        device = surface.get_xyz.device
        archive = self._device(device)
        sampled_numpy = self.sampler.next(maximum_anchors)
        sampled = torch.from_numpy(sampled_numpy).to(
            device=device, dtype=torch.long
        )
        self.anchor_rows_sampled += int(len(sampled_numpy))
        chosen = torch.zeros(
            len(self.xyz), dtype=torch.bool, device=device
        )
        chosen[sampled] = True
        candidate = (
            (surface._source_type == 2) & (surface._track_id < -1)
        )
        primitive_indices = torch.nonzero(
            candidate, as_tuple=False
        ).flatten()
        if not len(primitive_indices):
            return zero, {
                "matched": 0,
                "live_chart_rows": 0,
                "uv_edges_sampled": 0,
                "uv_edges_matched": 0,
                "coverage": self.sampler.audit(),
                "edge_coverage": self.edge_sampler.audit(),
            }
        primitive_ids = surface._track_id[primitive_indices]
        positions = torch.searchsorted(
            archive["sorted_evidence_id"], primitive_ids
        )
        in_range = positions < len(self.xyz)
        safe_positions = positions.clamp_max(max(len(self.xyz) - 1, 0))
        valid = in_range & (
            archive["sorted_evidence_id"][safe_positions]
            == primitive_ids
        )
        safe_rows = archive["evidence_id_order"][safe_positions]
        primitive_indices = primitive_indices[valid]
        rows = safe_rows[valid]
        if not len(primitive_indices):
            return zero, {
                "matched": 0,
                "live_chart_rows": 0,
                "uv_edges_sampled": 0,
                "uv_edges_matched": 0,
                "coverage": self.sampler.audit(),
                "edge_coverage": self.edge_sampler.audit(),
            }
        residual = (
            surface.get_xyz[primitive_indices] - archive["xyz"][rows]
        ).norm(dim=-1)
        anchor_mask = chosen[rows]
        if bool(anchor_mask.any()):
            anchor_rows = rows[anchor_mask]
            sigma = archive["footprint"][anchor_rows].clamp(0.004, 0.08)
            primitive_loss = torch.log1p(
                (residual[anchor_mask] / sigma).square()
            )
            unique_rows, inverse = torch.unique(
                anchor_rows, sorted=False, return_inverse=True
            )
            grouped = torch.full(
                (len(unique_rows),),
                float("inf"),
                device=device,
                dtype=primitive_loss.dtype,
            )
            # A Chart measurement needs one bootstrap witness.  This factor is
            # disabled before topology can replace that witness.
            grouped.scatter_reduce_(
                0, inverse, primitive_loss, reduce="amin", include_self=True
            )
            loss = grouped.mean()
        else:
            unique_rows = rows[:0]
            loss = zero
        # Preserve the differential structure of each Chart in UV space. This
        # is the discrete atlas Jacobian factor missing from the former
        # independent point-anchor implementation.  Endpoint resolution must
        # use every live Chart witness, not only the independently sampled
        # anchor minibatch.  The old coupling made an edge active only when
        # both endpoints happened to be selected as anchors in the same call;
        # with thousands of anchors that probability was effectively zero.
        edge_rows_numpy = self.edge_sampler.next(maximum_anchors)
        edge_loss = zero
        matched_edge_count = 0
        if len(edge_rows_numpy):
            edge_rows = torch.from_numpy(edge_rows_numpy).to(
                device=device, dtype=torch.long
            )
            edges = archive["edges"][edge_rows]
            # Chart primitives are not randomly split, so their negative
            # evidence id remains an exact row lookup.
            by_row = torch.full(
                (len(self.xyz),), -1, dtype=torch.long, device=device
            )
            best_residual = torch.full(
                (len(self.xyz),),
                float("inf"),
                dtype=residual.dtype,
                device=device,
            )
            best_residual.scatter_reduce_(
                0, rows, residual, reduce="amin", include_self=True
            )
            nearest = residual <= best_residual[rows] + 1e-8
            sentinel = torch.iinfo(torch.int64).max
            nearest_primitive = torch.where(
                nearest,
                primitive_indices,
                torch.full_like(primitive_indices, sentinel),
            )
            by_row.fill_(sentinel)
            by_row.scatter_reduce_(
                0,
                rows,
                nearest_primitive,
                reduce="amin",
                include_self=True,
            )
            by_row[by_row == sentinel] = -1
            pair = by_row[edges]
            valid_pair = (pair >= 0).all(dim=1)
            matched_edge_count = int(valid_pair.sum())
            if bool(valid_pair.any()):
                pair = pair[valid_pair]
                base = archive["xyz"][edges[valid_pair]]
                current_delta = (
                    surface.get_xyz[pair[:, 1]]
                    - surface.get_xyz[pair[:, 0]]
                )
                base_delta = base[:, 1] - base[:, 0]
                scale = base_delta.norm(dim=-1).clamp_min(0.005)
                edge_loss = torch.log1p(
                    (
                        (current_delta - base_delta).norm(dim=-1) / scale
                    ).square()
                ).mean()
        loss = loss + 0.25 * edge_loss
        self.anchor_rows_matched += int(len(unique_rows))
        self.edge_rows_sampled += int(len(edge_rows_numpy))
        self.edge_rows_matched += matched_edge_count
        if matched_edge_count:
            self.factor_calls_with_matched_edges += 1
        return loss, {
            "matched": int(len(unique_rows)),
            "live_chart_rows": int(torch.unique(rows).numel()),
            "uv_edges_sampled": int(len(edge_rows_numpy)),
            "uv_edges_matched": matched_edge_count,
            "coverage": self.sampler.audit(),
            "edge_coverage": self.edge_sampler.audit(),
        }

    def audit(self) -> dict:
        atlas_audit = (
            {
                "representation": "discrete_chart_observation_graph",
                "continuous_learnable_inverse_depth_atlas": False,
                "uv_domain_quadtree_densification": False,
                "renderer_binding": "bootstrap_only",
            }
            if self.atlas is None
            else self.atlas.audit()
        )
        return {
            **atlas_audit,
            "permanent_factor": (
                "source_resolution_rendered_inverse_depth_observation"
            ),
            "anchor_count": len(self.xyz),
            "coverage_only_seed_count": self.coverage_only_seed_count,
            "chart_count": int(len(np.unique(self.chart_id))),
            "chart_base_source": self.chart_base_source,
            "uv_bound": True,
            "uv_edge_count": int(len(self.edges)),
            "coverage": self.sampler.audit(),
            "edge_coverage": self.edge_sampler.audit(),
            "runtime": {
                "factor_calls": self.factor_calls,
                "anchor_rows_sampled": self.anchor_rows_sampled,
                "anchor_rows_matched": self.anchor_rows_matched,
                "edge_rows_sampled": self.edge_rows_sampled,
                "edge_rows_matched": self.edge_rows_matched,
                "edge_match_fraction": (
                    self.edge_rows_matched
                    / max(self.edge_rows_sampled, 1)
                ),
                "factor_calls_with_matched_edges": (
                    self.factor_calls_with_matched_edges
                ),
            },
        }

    def assert_complete_coverage(self) -> None:
        audit = self.sampler.audit()
        if audit["never_visited"]:
            raise RuntimeError(
                f"Chart evidence epoch did not reach full coverage: {audit}"
            )


class LearnableInverseDepthAtlas(torch.nn.Module):
    """Bounded residual inverse depth with UV-domain replace-and-retire."""

    maximum_relative_residual = 0.18

    def __init__(
        self,
        *,
        evidence_store: Path,
        seed_xyz: np.ndarray,
        evidence_id: np.ndarray,
        chart_id: np.ndarray,
        uv: np.ndarray,
        tangent_scale: np.ndarray,
        maximum_tangent_scale: float = float("inf"),
        chart_base_source: str = "matcha",
        payload: dict | None = None,
    ):
        super().__init__()
        if maximum_tangent_scale <= 0:
            raise ValueError("maximum_tangent_scale must be positive")
        self.maximum_tangent_scale = float(maximum_tangent_scale)
        if chart_base_source not in MOGE3_CHART_BASE_SOURCES:
            raise ValueError(
                f"Unsupported Chart base source: {chart_base_source!r}"
            )
        self.chart_base_source = str(chart_base_source)
        if len(evidence_id) and (
            np.any(evidence_id >= -1)
            or len(np.unique(evidence_id)) != len(evidence_id)
        ):
            raise RuntimeError(
                "Every Chart UV cell needs a unique evidence id below -1"
            )
        payload = (
            self._load_payload(
                evidence_store,
                chart_base_source=self.chart_base_source,
            )
            if payload is None
            else payload
        )
        self.register_buffer(
            "base_inverse_depth",
            torch.from_numpy(payload["inverse_depth"]),
        )
        self.register_buffer(
            "base_validity", torch.from_numpy(payload["validity"])
        )
        self.register_buffer(
            "camera_to_world",
            torch.from_numpy(payload["camera_to_world"]),
        )
        self.register_buffer(
            "intrinsics", torch.from_numpy(payload["intrinsics"])
        )
        chart_count = int(self.base_inverse_depth.shape[0])
        self.coarse_residual = torch.nn.Parameter(
            torch.zeros(chart_count, 1, 36, 64)
        )
        self.fine_residual = torch.nn.Parameter(
            torch.zeros(chart_count, 1, 72, 128)
        )
        self.evidence_id = evidence_id.astype(np.int64).copy()
        self.chart_id = chart_id.astype(np.int32).copy()
        self.uv = uv.astype(np.float32).copy()
        self.half_uv = self._initial_half_uv(chart_id)
        self.level = np.zeros(len(evidence_id), dtype=np.int16)
        self.base_rho = self._seed_base_rho(
            seed_xyz, chart_id, uv, payload["camera_to_world"]
        )
        if tangent_scale.shape != (len(evidence_id), 2):
            raise RuntimeError(
                "Chart cells require two source-UV tangent scales"
            )
        # This is the metric footprint of the source UV cell at base_rho,
        # not a free renderer scale.  At another inverse depth the same
        # angular cell scales by depth_current / depth_base. Quadtree
        # children halve this footprint explicitly.
        self.base_tangent_scale = (
            tangent_scale.astype(np.float32).copy()
        )
        self._initialize_residual_from_seed_consensus(chart_id, uv)
        self.next_evidence_id = (
            int(evidence_id.min()) - 1 if len(evidence_id) else -2
        )
        self.render_calls = 0
        self.rows_rendered = 0
        self.quadtree_events: list[dict] = []
        self._device_cache: dict[str, dict[str, torch.Tensor]] = {}

    @staticmethod
    def _artifact(manifest: dict, name: str) -> Path:
        for row in manifest.get("artifacts", []):
            if row.get("name") == name:
                return Path(row["path"])
        raise KeyError(f"Evidence store has no {name!r} artifact")

    @classmethod
    def _load_payload(
        cls,
        evidence_store: Path,
        *,
        chart_base_source: str = "matcha",
    ) -> dict:
        manifest_path = (
            evidence_store / "evidence_manifest.json"
            if evidence_store.is_dir()
            else evidence_store
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chart_path = cls._artifact(manifest, "chart_geometry")
        chart_camera_path = cls._artifact(manifest, "chart_cameras")
        with np.load(chart_path, allow_pickle=False) as archive:
            depth = archive["depths"].astype(np.float32)
            confidence = archive["confs"].astype(np.float32)
            scale = float(archive["scale_factor"])
            valid = (
                np.isfinite(depth)
                & (depth > 0)
                & np.isfinite(confidence)
                & (confidence > 0)
            )
            if "quality_selection_active" in archive:
                active = np.asarray(
                    archive["quality_selection_active"]
                ).astype(bool)
                if active.ndim == 1:
                    valid &= active[:, None, None]
            if "alignment_reference_mask" in archive:
                valid &= np.asarray(
                    archive["alignment_reference_mask"]
                ).astype(bool)
            inverse_depth = np.zeros_like(depth, dtype=np.float32)
            inverse_depth[valid] = scale / depth[valid]
        chart_cameras = json.loads(
            chart_camera_path.read_text(encoding="utf-8")
        )
        if chart_base_source != "matcha":
            chart_base_path = cls._artifact(manifest, "moge3_chart_base")
            base = load_chart_base(
                chart_base_path,
                source=chart_base_source,
                expected_names=chart_cameras["filepaths"],
            )
            if base["depth"].shape != depth.shape:
                raise RuntimeError(
                    "MoGe3 Chart base shape does not match MAtCha"
                )
            depth = base["depth"]
            valid = base["validity"] & np.isfinite(depth) & (depth > 0.0)
            inverse_depth = np.zeros_like(depth, dtype=np.float32)
            inverse_depth[valid] = 1.0 / depth[valid]
        scene = json.loads(
            Path(manifest["scene_contract"]).read_text(encoding="utf-8")
        )
        fixed_by_stem = {
            Path(row["image_name"]).stem: row
            for row in scene["records"]
        }
        h, w = depth.shape[1:]
        transforms = []
        intrinsics = []
        for filepath in chart_cameras["filepaths"]:
            stem = Path(filepath).stem
            if stem not in fixed_by_stem:
                raise RuntimeError(
                    f"Chart camera {stem!r} is absent from the fixed scene"
                )
            row = fixed_by_stem[stem]
            camera = row["camera"]
            sx = w / float(camera["width"])
            sy = h / float(camera["height"])
            transforms.append(
                np.asarray(row["T_camera_to_world"], dtype=np.float32)
            )
            intrinsics.append(
                (
                    float(camera["fx"]) * sx,
                    float(camera["fy"]) * sy,
                    float(camera["cx"]) * sx,
                    float(camera["cy"]) * sy,
                )
            )
        return {
            "inverse_depth": inverse_depth,
            "validity": valid.astype(np.float32),
            "camera_to_world": np.stack(transforms),
            "intrinsics": np.asarray(intrinsics, dtype=np.float32),
            "chart_base_source": str(chart_base_source),
        }

    @staticmethod
    def _seed_xyz_from_payload(
        payload: dict,
        chart_id: np.ndarray,
        uv: np.ndarray,
        fallback_xyz: np.ndarray,
    ) -> np.ndarray:
        """Move Chart bootstrap anchors onto the selected immutable base."""
        if not len(chart_id):
            return np.asarray(fallback_xyz, dtype=np.float32).copy()
        chart = torch.from_numpy(np.asarray(chart_id, dtype=np.int64))
        coordinates = torch.from_numpy(np.asarray(uv, dtype=np.float32))
        inverse = torch.from_numpy(payload["inverse_depth"])
        validity = torch.from_numpy(payload["validity"])
        rho = LearnableInverseDepthAtlas._bilinear(
            inverse, chart, coordinates
        )
        supported = LearnableInverseDepthAtlas._bilinear(
            validity, chart, coordinates
        ) > 0.25
        intrinsics = torch.from_numpy(payload["intrinsics"])[chart]
        height, width = inverse.shape[-2:]
        pixel_x = coordinates[:, 0] * width - 0.5
        pixel_y = coordinates[:, 1] * height - 0.5
        depth = rho.clamp_min(1.0e-8).reciprocal()
        camera = torch.stack(
            [
                (pixel_x - intrinsics[:, 2]) * depth / intrinsics[:, 0],
                (pixel_y - intrinsics[:, 3]) * depth / intrinsics[:, 1],
                depth,
            ],
            dim=-1,
        )
        transform = torch.from_numpy(payload["camera_to_world"])[chart]
        world = torch.bmm(
            transform[:, :3, :3], camera[:, :, None]
        ).squeeze(-1) + transform[:, :3, 3]
        fallback = torch.from_numpy(
            np.asarray(fallback_xyz, dtype=np.float32)
        )
        world = torch.where(supported[:, None], world, fallback)
        return world.numpy().astype(np.float32)

    @staticmethod
    def _initial_half_uv(chart_id: np.ndarray) -> np.ndarray:
        half = np.empty((len(chart_id), 2), dtype=np.float32)
        for chart in np.unique(chart_id):
            rows = np.flatnonzero(chart_id == chart)
            count = max(len(rows), 1)
            columns = max(
                int(round(np.sqrt(count * 16.0 / 9.0))), 1
            )
            row_count = max(int(round(count / columns)), 1)
            half[rows] = (0.5 / columns, 0.5 / row_count)
        return half

    @staticmethod
    def _nearest_numpy(
        values: np.ndarray, chart_id: np.ndarray, uv: np.ndarray
    ) -> np.ndarray:
        h, w = values.shape[1:]
        x = np.clip(
            np.rint(uv[:, 0] * w - 0.5), 0, w - 1
        ).astype(np.int64)
        y = np.clip(
            np.rint(uv[:, 1] * h - 0.5), 0, h - 1
        ).astype(np.int64)
        return values[chart_id, y, x]

    def _seed_base_rho(
        self,
        seed_xyz: np.ndarray,
        chart_id: np.ndarray,
        uv: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
        world_to_camera = np.linalg.inv(camera_to_world)
        rotation = world_to_camera[chart_id, :3, :3]
        translation = world_to_camera[chart_id, :3, 3]
        camera_xyz = (
            np.einsum("nij,nj->ni", rotation, seed_xyz) + translation
        )
        projected = 1.0 / np.clip(camera_xyz[:, 2], 1e-4, None)
        return projected.clip(1e-4, 10.0).astype(np.float32)

    @torch.no_grad()
    def _initialize_residual_from_seed_consensus(
        self, chart_id: np.ndarray, uv: np.ndarray
    ) -> None:
        """Warm-start the field without making seed XYZ a fixed target.

        The source atlas remains the immutable base.  Cross-view-consensus
        seed depths only initialize the learnable fine residual so switching
        a mature handoff to the atlas does not move supported facade cells by
        metres on iteration zero. Subsequent ray/RGB factors are free to
        change this state.
        """
        dense = self._nearest_numpy(
            self.base_inverse_depth.cpu().numpy(), chart_id, uv
        )
        valid = self._nearest_numpy(
            self.base_validity.cpu().numpy(), chart_id, uv
        ) > 0.25
        usable = (
            valid
            & np.isfinite(dense)
            & (dense > 1e-4)
            & np.isfinite(self.base_rho)
            & (self.base_rho > 1e-4)
        )
        if not bool(usable.any()):
            return
        relative = np.log(
            self.base_rho[usable] / dense[usable]
        ).clip(
            -0.95 * self.maximum_relative_residual,
            0.95 * self.maximum_relative_residual,
        )
        raw = np.arctanh(
            relative / self.maximum_relative_residual
        ).astype(np.float32)
        h, w = self.fine_residual.shape[-2:]
        x = np.clip(
            np.rint(uv[usable, 0] * w - 0.5), 0, w - 1
        ).astype(np.int64)
        y = np.clip(
            np.rint(uv[usable, 1] * h - 0.5), 0, h - 1
        ).astype(np.int64)
        chart = chart_id[usable].astype(np.int64)
        flat_index = (chart * h + y) * w + x
        total = np.zeros(
            self.fine_residual.numel(), dtype=np.float64
        )
        count = np.zeros_like(total)
        np.add.at(total, flat_index, raw)
        np.add.at(count, flat_index, 1.0)
        initialized = np.zeros_like(total, dtype=np.float32)
        selected = count > 0
        initialized[selected] = (
            total[selected] / count[selected]
        ).astype(np.float32)
        self.fine_residual.copy_(
            torch.from_numpy(initialized).reshape_as(
                self.fine_residual
            )
        )

    def _invalidate_cache(self) -> None:
        self._device_cache.clear()

    def _topology(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._device_cache:
            order = np.argsort(self.evidence_id)
            self._device_cache[key] = {
                "sorted_id": torch.from_numpy(
                    self.evidence_id[order]
                ).to(device=device),
                "order": torch.from_numpy(order).to(device=device),
                "chart_id": torch.from_numpy(self.chart_id).to(
                    device=device, dtype=torch.long
                ),
                "uv": torch.from_numpy(self.uv).to(device=device),
                "half_uv": torch.from_numpy(self.half_uv).to(device=device),
                "base_rho": torch.from_numpy(self.base_rho).to(device=device),
                "base_tangent_scale": torch.from_numpy(
                    self.base_tangent_scale
                ).to(device=device),
                "level": torch.from_numpy(self.level).to(
                    device=device, dtype=torch.int16
                ),
            }
        return self._device_cache[key]

    @staticmethod
    def _bilinear(
        values: torch.Tensor,
        chart_id: torch.Tensor,
        uv: torch.Tensor,
    ) -> torch.Tensor:
        """Sample [C,H,W] without materialising N copies of an atlas."""
        h, w = values.shape[-2:]
        x = (uv[:, 0] * w - 0.5).clamp(0, w - 1)
        y = (uv[:, 1] * h - 0.5).clamp(0, h - 1)
        x0 = x.floor().long()
        y0 = y.floor().long()
        x1 = (x0 + 1).clamp_max(w - 1)
        y1 = (y0 + 1).clamp_max(h - 1)
        wx = x - x0
        wy = y - y0
        a = values[chart_id, y0, x0]
        b = values[chart_id, y0, x1]
        c = values[chart_id, y1, x0]
        d = values[chart_id, y1, x1]
        return (
            a * (1 - wx) * (1 - wy)
            + b * wx * (1 - wy)
            + c * (1 - wx) * wy
            + d * wx * wy
        )

    def _rho(
        self,
        chart_id: torch.Tensor,
        uv: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        base = self._bilinear(self.base_inverse_depth, chart_id, uv)
        valid = self._bilinear(self.base_validity, chart_id, uv)
        coarse = self._bilinear(
            self.coarse_residual[:, 0], chart_id, uv
        )
        fine = self._bilinear(
            self.fine_residual[:, 0], chart_id, uv
        )
        base = torch.where(valid > 0.25, base, fallback)
        relative = self.maximum_relative_residual * torch.tanh(
            coarse + fine
        )
        return (base * torch.exp(relative)).clamp_min(1e-4)

    def _points(
        self,
        chart_id: torch.Tensor,
        uv: torch.Tensor,
        fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rho = self._rho(chart_id, uv, fallback)
        depth = rho.reciprocal()
        intrinsics = self.intrinsics[chart_id]
        h, w = self.base_inverse_depth.shape[-2:]
        pixel_x = uv[:, 0] * w - 0.5
        pixel_y = uv[:, 1] * h - 0.5
        camera = torch.stack(
            [
                (pixel_x - intrinsics[:, 2])
                / intrinsics[:, 0]
                * depth,
                (pixel_y - intrinsics[:, 3])
                / intrinsics[:, 1]
                * depth,
                depth,
            ],
            dim=-1,
        )
        transform = self.camera_to_world[chart_id]
        world = torch.bmm(
            transform[:, :3, :3], camera[:, :, None]
        ).squeeze(-1) + transform[:, :3, 3]
        return world, rho

    @staticmethod
    def _jacobian_frame(
        centre: torch.Tensor,
        right: torch.Tensor,
        down: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return metric half-cell scales and a complete UV tangent frame."""
        right_delta = right - centre
        down_delta = down - centre
        tangent_x = F.normalize(right_delta, dim=-1, eps=1e-6)
        normal = F.normalize(
            torch.linalg.cross(right_delta, down_delta, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        tangent_y = F.normalize(
            torch.linalg.cross(normal, tangent_x, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        rotation = torch.stack(
            [tangent_x, tangent_y, normal], dim=-1
        )
        # WXYZ conversion with a small differentiable floor.  Chart frames
        # are often close to identity; an exact sqrt(0) in the unused
        # quaternion components otherwise yields infinite atlas gradients.
        m = rotation
        qw = 0.5 * torch.sqrt(
            (
                1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
            ).clamp_min(1e-8)
        )
        qx = 0.5 * torch.sqrt(
            (
                1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]
            ).clamp_min(1e-8)
        )
        qy = 0.5 * torch.sqrt(
            (
                1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2]
            ).clamp_min(1e-8)
        )
        qz = 0.5 * torch.sqrt(
            (
                1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]
            ).clamp_min(1e-8)
        )
        quaternion = torch.stack(
            [
                qw,
                qx
                * torch.sign(m[:, 2, 1] - m[:, 1, 2] + 1e-12),
                qy
                * torch.sign(m[:, 0, 2] - m[:, 2, 0] + 1e-12),
                qz
                * torch.sign(m[:, 1, 0] - m[:, 0, 1] + 1e-12),
            ],
            dim=-1,
        )
        scales = torch.stack(
            [
                right_delta.norm(dim=-1),
                # The second renderer tangent is the orthogonal component
                # of the UV-v Jacobian, not the diagonal length that also
                # contains UV-u shear.
                (down_delta * tangent_y).sum(-1).abs(),
            ],
            dim=-1,
        ).clamp_min(1e-6)
        return scales, F.normalize(
            quaternion, dim=-1, eps=1e-6
        )

    def _local_chart_geometry(
        self,
        chart_id: torch.Tensor,
        uv: torch.Tensor,
        half: torch.Tensor,
        fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a discontinuity-safe local Chart tangent frame.

        A depth atlas is a collection of surface samples, not a mesh across
        every image-space depth discontinuity.  The former implementation
        always differentiated towards positive ``u``/``v`` (except at the
        image boundary).  Consequently a cell immediately to the left of a
        foreground/background edge used the point on the other depth layer
        as its tangent endpoint.  The resulting half-cell could be metres
        long, was merely clamped to the global 0.5 m safety ceiling, and
        rendered as the characteristic diagonal/needle paint stroke.

        Both one-sided derivatives are observable in the source Chart.  Use
        the side whose inverse depth is closest to the centre, then orient a
        backward difference back into the positive UV convention.  Smooth
        oblique surfaces retain their metric depth derivative; an actual
        discontinuity is no longer bridged by a surfel.
        """
        centre, centre_rho = self._points(chart_id, uv, fallback)

        def one_axis(axis: int) -> torch.Tensor:
            plus_uv = uv.clone()
            minus_uv = uv.clone()
            plus_uv[:, axis] = (uv[:, axis] + half[:, axis]).clamp(0, 1)
            minus_uv[:, axis] = (uv[:, axis] - half[:, axis]).clamp(0, 1)
            plus, plus_rho = self._points(
                chart_id, plus_uv, fallback
            )
            minus, minus_rho = self._points(
                chart_id, minus_uv, fallback
            )
            plus_valid = plus_uv[:, axis] > uv[:, axis]
            minus_valid = minus_uv[:, axis] < uv[:, axis]
            centre_log_rho = centre_rho.clamp_min(1e-8).log()
            plus_cost = (
                plus_rho.clamp_min(1e-8).log() - centre_log_rho
            ).abs()
            minus_cost = (
                minus_rho.clamp_min(1e-8).log() - centre_log_rho
            ).abs()
            infinity = torch.full_like(plus_cost, float("inf"))
            plus_cost = torch.where(plus_valid, plus_cost, infinity)
            minus_cost = torch.where(minus_valid, minus_cost, infinity)
            choose_plus = plus_cost <= minus_cost
            endpoint = torch.where(choose_plus[:, None], plus, minus)
            sign = torch.where(
                choose_plus,
                torch.ones_like(plus_cost),
                -torch.ones_like(plus_cost),
            )
            return centre + (endpoint - centre) * sign[:, None]

        right = one_axis(0)
        down = one_axis(1)
        scales, quaternion = self._jacobian_frame(
            centre, right, down
        )
        return centre, scales, quaternion

    def _live_rows(self, surface):
        device = surface.get_xyz.device
        topology = self._topology(device)
        candidates = torch.nonzero(
            (surface._source_type == 2) & (surface._track_id < -1),
            as_tuple=False,
        ).flatten()
        if not len(candidates) or not len(self.evidence_id):
            return candidates[:0], candidates[:0], topology
        ids = surface._track_id[candidates]
        positions = torch.searchsorted(topology["sorted_id"], ids)
        in_range = positions < len(self.evidence_id)
        safe = positions.clamp_max(max(len(self.evidence_id) - 1, 0))
        valid = in_range & (topology["sorted_id"][safe] == ids)
        return (
            candidates[valid],
            topology["order"][safe[valid]],
            topology,
        )

    def live_surface_rows(self, surface) -> torch.Tensor:
        """Public row-order contract for renderer footprint feedback."""
        primitive, _, _ = self._live_rows(surface)
        return primitive

    def _bound_geometry(self, surface):
        primitive, atlas_row, topology = self._live_rows(surface)
        if not len(primitive):
            return primitive, atlas_row, None
        chart_id = topology["chart_id"][atlas_row]
        uv = topology["uv"][atlas_row]
        half = topology["half_uv"][atlas_row]
        fallback = topology["base_rho"][atlas_row]
        centre, scales, quaternion = (
            LearnableInverseDepthAtlas._local_chart_geometry(
                self, chart_id, uv, half, fallback
            )
        )
        # The atlas remains geometrically trainable after a validated rigid
        # handoff, including when free surface xyz/scale/rotation are frozen
        # by ``appearance_only``.  Consequently the trainer's physical
        # surface-scale contract must be applied here, on the actual geometry
        # sent to the renderer.  Clamping only ``surface._scaling`` is
        # ineffective because this override replaces that tensor every
        # render.  It previously allowed a small number of sparse/far Chart
        # cells to reach tens of metres and become low-frequency paint
        # layers.
        scales = scales.clamp_max(self.maximum_tangent_scale)
        # ``surface._scaling`` is the persistent per-cell bandwidth ceiling.
        # Screen-space repair is written there after the rasterizer reports
        # an oversized footprint.  Ignoring it here made every such repair a
        # one-step no-op because the next live-atlas forward replaced the
        # repaired scale with the raw Jacobian again.  Taking the minimum
        # keeps the atlas authoritative for depth/orientation while allowing
        # measured multi-view screen evidence to impose a durable ceiling.
        persistent_scale_ceiling = getattr(surface, "get_scaling", None)
        if persistent_scale_ceiling is not None:
            persistent_scale_ceiling = persistent_scale_ceiling[primitive]
            if persistent_scale_ceiling.shape != scales.shape:
                raise RuntimeError(
                    "Chart surface scale ceiling no longer aligns with "
                    "live atlas rows"
                )
            scales = torch.minimum(scales, persistent_scale_ceiling)
        return primitive, atlas_row, (centre, scales, quaternion)

    def geometry_override(self, surface):
        primitive, _, geometry = self._bound_geometry(surface)
        if geometry is None:
            return None
        centre, scales, quaternion = geometry
        means = surface.get_xyz.clone()
        tangent = surface.get_scaling.clone()
        rotations = surface.get_rotation.clone()
        means[primitive] = centre
        tangent[primitive] = scales
        rotations[primitive] = quaternion
        self.render_calls += 1
        self.rows_rendered += int(len(primitive))
        return means, tangent, rotations

    def regularizer(self) -> torch.Tensor:
        magnitude = (
            self.coarse_residual.square().mean()
            + 0.5 * self.fine_residual.square().mean()
        )
        smooth = (
            (self.coarse_residual[..., 1:, :] - self.coarse_residual[..., :-1, :])
            .abs()
            .mean()
            + (self.coarse_residual[..., :, 1:] - self.coarse_residual[..., :, :-1])
            .abs()
            .mean()
            + 0.5
            * (
                self.fine_residual[..., 1:, :]
                - self.fine_residual[..., :-1, :]
            )
            .abs()
            .mean()
            + 0.5
            * (
                self.fine_residual[..., :, 1:]
                - self.fine_residual[..., :, :-1]
            )
            .abs()
            .mean()
        )
        return magnitude + 0.10 * smooth

    @torch.no_grad()
    def adapt_uv_quadtree(
        self,
        surface,
        *,
        maximum_net_growth: int,
        maximum_points: int,
        gradient_threshold: float,
        radius_target: float,
        maximum_level: int = 3,
        priority_snapshot: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        if int(maximum_level) < 0:
            raise ValueError("maximum_level cannot be negative")
        maximum_net_growth = min(
            int(maximum_net_growth),
            max(int(maximum_points) - len(surface.get_xyz), 0),
        )
        parent_capacity = maximum_net_growth // 3
        primitive, atlas_row, geometry = self._bound_geometry(surface)
        if parent_capacity <= 0 or geometry is None:
            return {
                "enabled": True,
                "parents_replaced": 0,
                "children": 0,
                "net_growth": 0,
                "configured_maximum_level": int(maximum_level),
            }
        if priority_snapshot is None:
            priority_snapshot = self.topology_priority_snapshot(
                surface,
                gradient_threshold=gradient_threshold,
                radius_target=radius_target,
            )
        snapshot_id = priority_snapshot["evidence_id"].to(
            device=primitive.device
        )
        snapshot_priority = priority_snapshot["priority"].to(
            device=primitive.device
        )
        if not len(snapshot_id):
            return {
                "enabled": True,
                "parents_replaced": 0,
                "children": 0,
                "net_growth": 0,
                "eligible": 0,
                "configured_maximum_level": int(maximum_level),
            }
        order = torch.argsort(snapshot_id)
        sorted_id = snapshot_id[order]
        current_id = surface._track_id[primitive]
        positions = torch.searchsorted(sorted_id, current_id)
        safe = positions.clamp_max(max(len(sorted_id) - 1, 0))
        matched = (
            (positions < len(sorted_id))
            & (sorted_id[safe] == current_id)
        )
        priority = torch.zeros(
            len(primitive), device=primitive.device
        )
        priority[matched] = snapshot_priority[order[safe[matched]]]
        topology = self._topology(surface.get_xyz.device)
        # The fine inverse-depth field is 128x72 and the Cambridge training
        # target is 640x360.  Three binary subdivisions already reach the
        # target's observable pixel bandwidth.  Repeatedly splitting the same
        # lineage to level 10 created sub-pixel duplicate surfels without any
        # additional depth or RGB evidence and consumed the surface budget
        # needed by uncovered facade cells.
        eligible = (priority >= 1.0) & (
            topology["level"][atlas_row] < int(maximum_level)
        )
        eligible_indices = torch.nonzero(
            eligible, as_tuple=False
        ).flatten()
        count = min(parent_capacity, len(eligible_indices))
        if count <= 0:
            return {
                "enabled": True,
                "parents_replaced": 0,
                "children": 0,
                "net_growth": 0,
                "eligible": int(len(eligible_indices)),
                "configured_maximum_level": int(maximum_level),
            }
        _, order = torch.topk(priority[eligible_indices], k=count)
        selected_local = eligible_indices[order]
        parents = primitive[selected_local]
        parent_atlas = atlas_row[selected_local]
        parent_uv = topology["uv"][parent_atlas]
        parent_half = topology["half_uv"][parent_atlas]
        parent_chart = topology["chart_id"][parent_atlas]
        parent_base_rho = topology["base_rho"][parent_atlas]
        offsets = parent_half[:, None, :] * parent_uv.new_tensor(
            ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))
        )[None]
        child_uv = (parent_uv[:, None, :] + offsets).clamp(0, 1).reshape(
            -1, 2
        )
        child_chart = parent_chart.repeat_interleave(4)
        child_fallback = parent_base_rho.repeat_interleave(4)
        child_half = parent_half.repeat_interleave(4, dim=0) * 0.5
        (
            child_xyz,
            child_scale,
            child_rotation,
        ) = LearnableInverseDepthAtlas._local_chart_geometry(
            self,
            child_chart,
            child_uv,
            child_half,
            child_fallback,
        )
        child_scale = child_scale.clamp_max(
            float(
                getattr(
                    self, "maximum_tangent_scale", float("inf")
                )
            )
        )
        # Retain the metric Jacobian scale in the topology archive for
        # checkpoint/export compatibility.  Runtime geometry always
        # recomputes it from the current learned atlas.
        child_base_tangent_scale = child_scale.detach()
        repeated = parents.repeat_interleave(4)
        child_scaling = child_scale.clamp_min(1e-6).log()
        child_ids = torch.arange(
            self.next_evidence_id,
            self.next_evidence_id - 4 * count,
            -1,
            device=parents.device,
            dtype=torch.int64,
        )
        metadata = surface._point_metadata_from_indices(
            parents,
            repeat=4,
            source_type=2,
            track_id=-1,
            protected_flag=False,
        )
        metadata["track_id"] = child_ids
        surface.densification_postfix(
            child_xyz,
            surface._features_dc[repeated],
            surface._features_rest[repeated],
            surface._opacity[repeated],
            child_scaling,
            child_rotation,
            metadata=metadata,
        )
        remove = torch.zeros(
            len(surface.get_xyz), dtype=torch.bool, device=parents.device
        )
        remove[parents] = True
        surface.prune_points(remove)

        selected_cpu = parent_atlas.cpu().numpy()
        self.evidence_id = np.concatenate(
            [self.evidence_id, child_ids.cpu().numpy()]
        )
        self.chart_id = np.concatenate(
            [
                self.chart_id,
                self.chart_id[selected_cpu].repeat(4),
            ]
        )
        self.uv = np.concatenate(
            [self.uv, child_uv.cpu().numpy().astype(np.float32)]
        )
        self.half_uv = np.concatenate(
            [
                self.half_uv,
                child_half.cpu().numpy().astype(np.float32),
            ]
        )
        self.base_rho = np.concatenate(
            [
                self.base_rho,
                self.base_rho[selected_cpu].repeat(4),
            ]
        )
        self.base_tangent_scale = np.concatenate(
            [
                self.base_tangent_scale,
                child_base_tangent_scale.cpu().numpy().astype(
                    np.float32
                ),
            ]
        )
        self.level = np.concatenate(
            [
                self.level,
                (self.level[selected_cpu] + 1).repeat(4),
            ]
        )
        self.next_evidence_id -= 4 * count
        self._invalidate_cache()
        event = {
            "enabled": True,
            "parents_replaced": int(count),
            "children": int(4 * count),
            "net_growth": int(3 * count),
            "eligible": int(len(eligible_indices)),
            "mean_parent_priority": float(
                priority[selected_local].mean()
            ),
            "maximum_level": int(self.level.max()),
            "configured_maximum_level": int(maximum_level),
            "priority_audit": {
                name: float(priority_snapshot[name])
                for name in (
                    "gradient_normalizer",
                    "raw_gradient_p50",
                    "raw_gradient_p95",
                    "raw_gradient_maximum",
                    "raw_radius_p95",
                    "raw_radius_maximum",
                )
                if name in priority_snapshot
            },
        }
        self.quadtree_events.append(event)
        return event

    @torch.no_grad()
    def topology_priority_snapshot(
        self,
        surface,
        *,
        gradient_threshold: float,
        radius_target: float,
    ) -> dict[str, torch.Tensor]:
        primitive, _, _ = self._bound_geometry(surface)
        if not len(primitive):
            return {
                "evidence_id": torch.empty(0, dtype=torch.int64),
                "priority": torch.empty(0),
            }
        grads = torch.nan_to_num(
            surface.xyz_gradient_accum[primitive]
            / surface.denom[primitive].clamp_min(1e-8),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).norm(dim=-1)
        radii = surface.max_radii2D[primitive].float()
        confidence = surface._geometry_confidence[primitive].float()
        reliability = 0.4 + 0.6 * confidence / (confidence + 0.25)
        positive_gradient = grads[torch.isfinite(grads) & (grads > 0)]
        gradient_normalizer = grads.new_tensor(
            max(float(gradient_threshold), 1e-12)
        )
        if len(positive_gradient):
            gradient_normalizer = torch.maximum(
                gradient_normalizer,
                torch.quantile(positive_gradient, 0.75),
            )
        # Screen-coordinate gradients are renderer-unit dependent and may
        # contain a few enormous near-plane values. Their old raw
        # ``grad/threshold`` score reached ~1e12 on a real Cambridge event,
        # so top-k became an outlier selector rather than an error selector.
        # Robust population normalization preserves ordering while bounding
        # either evidence term's influence.
        gradient_score = (
            grads / gradient_normalizer.clamp_min(1e-12)
        ).clamp(0.0, 8.0)
        radius_score = (
            radii / max(float(radius_target), 1e-6)
        ).square().clamp(0.0, 8.0)
        # A quadtree split buys *area bandwidth*, not another vote for the
        # same isolated high-gradient sample.  The former pointwise gradient
        # term stayed at its maximum after a child became sub-pixel, so one
        # hot lineage could be selected again at every 100-step event (the
        # real Cambridge run reached level 19 while >70k coarse cells still
        # needed bandwidth).  Integrating the gradient density over the
        # projected cell area makes the score approximate the marginal image
        # error a split can actually reduce.  Radius demand remains an
        # independent term, while a sub-pixel child naturally yields to a
        # coarser unresolved cell without an arbitrary maximum-level gate.
        projected_area_weight = (
            radii / max(float(radius_target), 1e-6)
        ).square().clamp(0.0, 1.0)
        priority = reliability * (
            gradient_score * projected_area_weight + radius_score
        )
        priority = torch.where(
            radii > 0, priority, torch.zeros_like(priority)
        )
        def quantile(value: torch.Tensor, fraction: float) -> torch.Tensor:
            finite = value[torch.isfinite(value)]
            return (
                torch.quantile(finite, fraction)
                if len(finite)
                else value.new_zeros(())
            )

        return {
            "evidence_id": surface._track_id[primitive].detach().cpu(),
            "priority": priority.detach().cpu(),
            "gradient_normalizer": gradient_normalizer.detach().cpu(),
            "raw_gradient_p50": quantile(grads, 0.50).detach().cpu(),
            "raw_gradient_p95": quantile(grads, 0.95).detach().cpu(),
            "raw_gradient_maximum": quantile(
                grads, 1.0
            ).detach().cpu(),
            "raw_radius_p95": quantile(radii, 0.95).detach().cpu(),
            "raw_radius_maximum": quantile(
                radii, 1.0
            ).detach().cpu(),
        }

    @torch.no_grad()
    def bake_into_surface(self, surface) -> dict:
        primitive, _, geometry = self._bound_geometry(surface)
        if geometry is None:
            return {"baked_rows": 0}
        centre, scales, quaternion = geometry
        surface._xyz[primitive] = centre
        surface._scaling[primitive] = scales.clamp_min(1e-6).log()
        surface._rotation[primitive] = quaternion
        return {"baked_rows": int(len(primitive))}

    def capture(self) -> dict:
        return {
            "version": "learnable-inverse-depth-atlas-v2",
            "residual_state": {
                "coarse_residual": self.coarse_residual.detach().cpu(),
                "fine_residual": self.fine_residual.detach().cpu(),
            },
            "evidence_id": torch.from_numpy(self.evidence_id),
            "chart_id": torch.from_numpy(self.chart_id),
            "uv": torch.from_numpy(self.uv),
            "half_uv": torch.from_numpy(self.half_uv),
            "base_rho": torch.from_numpy(self.base_rho),
            "base_tangent_scale": torch.from_numpy(
                self.base_tangent_scale
            ),
            "level": torch.from_numpy(self.level),
            "next_evidence_id": int(self.next_evidence_id),
            "render_calls": int(self.render_calls),
            "rows_rendered": int(self.rows_rendered),
            "quadtree_events": self.quadtree_events,
        }

    def restore(self, state: dict) -> None:
        if state.get("version") not in {
            "learnable-inverse-depth-atlas-v1",
            "learnable-inverse-depth-atlas-v2",
        }:
            raise RuntimeError("Unsupported inverse-depth atlas checkpoint")
        device = self.coarse_residual.device
        residual_state = state["residual_state"]
        with torch.no_grad():
            self.coarse_residual.copy_(
                torch.as_tensor(
                    residual_state["coarse_residual"], device=device
                )
            )
            self.fine_residual.copy_(
                torch.as_tensor(
                    residual_state["fine_residual"], device=device
                )
            )
        self.evidence_id = torch.as_tensor(
            state["evidence_id"], dtype=torch.int64
        ).cpu().numpy()
        self.chart_id = torch.as_tensor(
            state["chart_id"], dtype=torch.int32
        ).cpu().numpy()
        self.uv = torch.as_tensor(
            state["uv"], dtype=torch.float32
        ).cpu().numpy()
        self.half_uv = torch.as_tensor(
            state["half_uv"], dtype=torch.float32
        ).cpu().numpy()
        self.base_rho = torch.as_tensor(
            state["base_rho"], dtype=torch.float32
        ).cpu().numpy()
        if state.get("base_tangent_scale") is not None:
            self.base_tangent_scale = torch.as_tensor(
                state["base_tangent_scale"], dtype=torch.float32
            ).cpu().numpy()
        elif len(self.base_tangent_scale) != len(self.base_rho):
            raise RuntimeError(
                "Legacy Chart atlas topology cannot recover tangent scales"
            )
        self.level = torch.as_tensor(
            state["level"], dtype=torch.int16
        ).cpu().numpy()
        lengths = {
            len(self.evidence_id),
            len(self.chart_id),
            len(self.uv),
            len(self.half_uv),
            len(self.base_rho),
            len(self.base_tangent_scale),
            len(self.level),
        }
        if len(lengths) != 1:
            raise RuntimeError("Atlas topology arrays have unequal lengths")
        self.next_evidence_id = int(state["next_evidence_id"])
        self.render_calls = int(state.get("render_calls", 0))
        self.rows_rendered = int(state.get("rows_rendered", 0))
        self.quadtree_events = list(state.get("quadtree_events", []))
        self._invalidate_cache()

    def audit(self) -> dict:
        return {
            "representation": "learnable_bounded_inverse_depth_atlas",
            "continuous_learnable_inverse_depth_atlas": True,
            "uv_domain_quadtree_densification": True,
            "renderer_binding": "native_2d_surfel_geometry_override",
            "tangent_scale_contract": (
                "exact_k_current_inverse_depth_uv_jacobian_half_cell_"
                "physical_scale_bounded"
            ),
            "maximum_tangent_scale": self.maximum_tangent_scale,
            "rotation_contract": (
                "complete_orthonormal_uv_jacobian_frame_wxyz"
            ),
            "parameter_count": int(
                self.coarse_residual.numel() + self.fine_residual.numel()
            ),
            "topology_cell_count": int(len(self.evidence_id)),
            "maximum_level": int(self.level.max()) if len(self.level) else 0,
            "render_calls": int(self.render_calls),
            "rows_rendered": int(self.rows_rendered),
            "quadtree_event_count": len(self.quadtree_events),
            "final_export": "baked_standard_2dgs_geometry",
        }
