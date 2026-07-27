"""Permanent Chart-surface evidence independent of render topology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from outdoor.training_evidence import EvidenceEpochSampler


class ChartSurfaceModel:
    """UV-atlas geometry factor independent of render primitive topology.

    The Gaussian renderer may clone, split, or retire residual primitives.
    This object keeps the original Chart measurement alive and groups every
    descendant by its negative evidence id, preventing topology from changing
    the factor weight.
    """

    def __init__(self, surface_seed: Path, *, seed: int = 3253):
        with np.load(surface_seed, allow_pickle=False) as archive:
            source = archive["source_type"].astype(np.int8)
            selected = source == 2
            self.xyz = archive["xyz"][selected].astype(np.float32)
            self.sigma = archive["position_sigma"][selected].astype(np.float32)
            self.footprint = archive["scales"][selected].max(axis=1).astype(
                np.float32
            )
            self.evidence_id = archive["track_id"][selected].astype(np.int64)
            self.chart_id = archive["chart_id"][selected].astype(np.int32)
            self.uv = archive["chart_uv"][selected].astype(np.float32)
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
        self._cache = {}

    def _device(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = {
                "xyz": torch.from_numpy(self.xyz).to(device),
                "sigma": torch.from_numpy(self.sigma).to(device),
                "footprint": torch.from_numpy(self.footprint).to(device),
                "evidence_id": torch.from_numpy(self.evidence_id).to(device),
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
        # ids are -(row + 2), so lookup is exact and O(N).
        rows = -surface._track_id[primitive_indices] - 2
        valid = (rows >= 0) & (rows < len(self.xyz))
        safe_rows = rows.clamp(0, max(len(self.xyz) - 1, 0))
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
        grouped = torch.zeros(
            len(unique_rows), device=device, dtype=primitive_loss.dtype
        )
        counts = torch.zeros_like(grouped)
        grouped.scatter_add_(0, inverse, primitive_loss)
        counts.scatter_add_(0, inverse, torch.ones_like(primitive_loss))
        loss = (grouped / counts.clamp_min(1)).mean()
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
            by_row[rows] = primitive_indices
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
            "anchor_count": len(self.xyz),
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
