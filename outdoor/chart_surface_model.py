"""Permanent Chart-surface evidence independent of render topology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from outdoor.training_evidence import EvidenceEpochSampler


class ChartSurfaceModel:
    """Immutable Chart evidence plus a short-lived renderer-seed prior.

    The permanent source-resolution Chart depth factor is evaluated by
    ``OutdoorGeometryEvidence.chart_native_factor`` and never depends on a
    Gaussian identity.  This class retains UV/adjacency metadata and supplies
    only the bootstrap centre/Jacobian prior before renderer witnesses are
    released for normal local replacement.
    """

    def __init__(self, surface_seed: Path, *, seed: int = 3253):
        with np.load(surface_seed, allow_pickle=False) as archive:
            source = archive["source_type"].astype(np.int8)
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
        zero = surface.get_xyz.new_zeros(())
        if not len(self.xyz):
            return zero, {"matched": 0, "coverage": self.sampler.audit()}
        device = surface.get_xyz.device
        archive = self._device(device)
        sampled_numpy = self.sampler.next(maximum_anchors)
        sampled = torch.from_numpy(sampled_numpy).to(
            device=device, dtype=torch.long
        )
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
            return zero, {"matched": 0, "coverage": self.sampler.audit()}
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
        valid &= chosen[safe_rows]
        primitive_indices = primitive_indices[valid]
        rows = safe_rows[valid]
        if not len(primitive_indices):
            return zero, {"matched": 0, "coverage": self.sampler.audit()}
        sigma = archive["footprint"][rows].clamp(0.004, 0.08)
        residual = (
            surface.get_xyz[primitive_indices] - archive["xyz"][rows]
        ).norm(dim=-1)
        primitive_loss = torch.log1p((residual / sigma).square())
        unique_rows, inverse = torch.unique(
            rows, sorted=False, return_inverse=True
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
        # Preserve the differential structure of each Chart in UV space. This
        # is the discrete atlas Jacobian factor missing from the former
        # independent point-anchor implementation.
        edge_rows_numpy = self.edge_sampler.next(maximum_anchors)
        edge_loss = zero
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
            by_row[rows[nearest]] = primitive_indices[nearest]
            pair = by_row[edges]
            valid_pair = (pair >= 0).all(dim=1)
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
        return loss, {
            "matched": int(len(unique_rows)),
            "uv_edges": int(len(edge_rows_numpy)),
            "coverage": self.sampler.audit(),
        }

    def audit(self) -> dict:
        return {
            # Be explicit about the remaining architectural gap.  UV-bound
            # samples plus a local Jacobian prior are materially stronger than
            # independent XYZ anchors, but they are not MAtCha's learnable
            # inverse-depth atlas and must never be reported as such.
            "representation": "discrete_chart_observation_graph",
            "continuous_learnable_inverse_depth_atlas": False,
            "uv_domain_quadtree_densification": False,
            "renderer_seed_binding": "bootstrap_only",
            "permanent_factor": (
                "source_resolution_rendered_inverse_depth_observation"
            ),
            "anchor_count": len(self.xyz),
            "coverage_only_seed_count": self.coverage_only_seed_count,
            "chart_count": int(len(np.unique(self.chart_id))),
            "uv_bound": True,
            "uv_edge_count": int(len(self.edges)),
            "coverage": self.sampler.audit(),
        }

    def assert_complete_coverage(self) -> None:
        audit = self.sampler.audit()
        if audit["never_visited"]:
            raise RuntimeError(
                f"Chart evidence epoch did not reach full coverage: {audit}"
            )
