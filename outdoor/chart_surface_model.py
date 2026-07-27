"""Permanent Chart-surface evidence independent of render topology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from outdoor.training_evidence import EvidenceEpochSampler


class ChartSurfaceModel:
    """Chart-bound metric anchors with UV identity and epoch coverage.

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
        self._cache = {}

    def _device(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = {
                "xyz": torch.from_numpy(self.xyz).to(device),
                "sigma": torch.from_numpy(self.sigma).to(device),
                "evidence_id": torch.from_numpy(self.evidence_id).to(device),
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
        sigma = archive["sigma"][rows].clamp(0.01, 0.20)
        residual = (
            surface.get_xyz[primitive_indices] - archive["xyz"][rows]
        ).norm(dim=-1)
        # Descendants may occupy their UV cell instead of collapsing back to
        # the parent sample. Only motion beyond the metric cell uncertainty is
        # charged to the permanent Chart factor.
        primitive_loss = torch.log1p(
            ((residual - 0.75 * sigma).clamp_min(0) / sigma).square()
        )
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
        return loss, {
            "matched": int(len(unique_rows)),
            "coverage": self.sampler.audit(),
        }

    def audit(self) -> dict:
        return {
            "anchor_count": len(self.xyz),
            "chart_count": int(len(np.unique(self.chart_id))),
            "uv_bound": True,
            "coverage": self.sampler.audit(),
        }

    def assert_complete_coverage(self) -> None:
        audit = self.sampler.audit()
        if audit["never_visited"]:
            raise RuntimeError(
                f"Chart evidence epoch did not reach full coverage: {audit}"
            )
