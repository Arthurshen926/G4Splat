"""Lazy per-view geometry sources consumed by the unified teacher."""

from __future__ import annotations

from collections import OrderedDict
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from outdoor.evidence_store import (
    artifact_path,
    load_evidence_store,
    mast3r_is_geometry_authority,
)
from outdoor.inverse_depth import INVERSE_DEPTH_FUSION_VERSION
from outdoor.mast3r_track_graph import TRACK_GRAPH_VERSION
from outdoor.moge3_evidence import (
    load_index as load_moge3_index,
    load_runtime_cache as load_moge3_runtime_cache,
    load_view as load_moge3_view,
    sha256_file as moge3_sha256_file,
)
from outdoor.moge3_chart_base import (
    MOGE3_CHART_BASE_SOURCES,
    load_chart_base,
    load_chart_base_metadata,
)
from outdoor.role_aware_initialization import (
    SINGLE_SEQUENCE_POINTMAP_PRECISION,
    _load_mast3r_pointmap_geometry,
    _native_pointmap_shape,
)


class EvidenceEpochSampler:
    """Deterministic shuffled epochs with an auditable coverage contract."""

    def __init__(self, count: int, *, seed: int):
        self.count = int(count)
        self.rng = np.random.default_rng(int(seed))
        self.order = self.rng.permutation(self.count)
        self.cursor = 0
        self.epoch = 0
        self.visits = np.zeros(self.count, dtype=np.int64)

    def next(self, maximum: int) -> np.ndarray:
        if self.count == 0 or int(maximum) <= 0:
            return np.empty(0, dtype=np.int64)
        # Do not wrap into the next epoch inside one factor call.  Camera
        # tables have very different sizes; filling every tail batch back to
        # ``maximum`` used the remainder on duplicate rows from small tables.
        # The global capacity calculation then claimed a complete first pass
        # even though those duplicate slots left large-camera rows untouched.
        # A short tail batch is loss-normalized by its real row count, while
        # the outer least-progress scheduler can immediately advance the next
        # incomplete camera.  This makes ``completed_epochs`` an actual
        # evidence boundary rather than a boundary crossed mid-batch.
        available = self.count - self.cursor
        take = min(int(maximum), available)
        chunk = self.order[self.cursor : self.cursor + take]
        self.visits[chunk] += 1
        self.cursor += take
        if self.cursor == self.count:
            self.epoch += 1
            self.order = self.rng.permutation(self.count)
            self.cursor = 0
        return chunk.copy()

    def audit(self) -> dict:
        visited = int((self.visits > 0).sum())
        return {
            "count": self.count,
            "visited": visited,
            "never_visited": self.count - visited,
            "coverage": visited / max(self.count, 1),
            "completed_epochs": self.epoch,
            "minimum_visits": int(self.visits.min()) if self.count else 0,
            "maximum_visits": int(self.visits.max()) if self.count else 0,
        }

    def capture_state(self) -> dict:
        """Return the exact shuffled-epoch continuation state."""
        return {
            "version": "evidence-epoch-sampler-v1",
            "count": self.count,
            "order": torch.from_numpy(self.order.copy()),
            "cursor": int(self.cursor),
            "epoch": int(self.epoch),
            "visits": torch.from_numpy(self.visits.copy()),
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_state(self, state: dict) -> None:
        if state.get("version") != "evidence-epoch-sampler-v1":
            raise RuntimeError(
                "Unsupported evidence epoch sampler state: "
                f"{state.get('version')!r}"
            )
        if int(state.get("count", -1)) != self.count:
            raise RuntimeError(
                "Evidence epoch sampler count changed across resume"
            )
        order = torch.as_tensor(
            state["order"], dtype=torch.int64
        ).cpu().numpy()
        visits = torch.as_tensor(
            state["visits"], dtype=torch.int64
        ).cpu().numpy()
        if order.shape != (self.count,) or visits.shape != (self.count,):
            raise RuntimeError(
                "Evidence epoch sampler arrays have an invalid shape"
            )
        if self.count and not np.array_equal(
            np.sort(order), np.arange(self.count)
        ):
            raise RuntimeError(
                "Evidence epoch sampler order is not a permutation"
            )
        cursor = int(state.get("cursor", -1))
        epoch = int(state.get("epoch", -1))
        if (
            cursor < 0
            or cursor >= max(self.count, 1)
            or epoch < 0
            or bool((visits < 0).any())
        ):
            raise RuntimeError(
                "Evidence epoch sampler counters are invalid"
            )
        self.order = order.copy()
        self.cursor = cursor
        self.epoch = epoch
        self.visits = visits.copy()
        self.rng.bit_generator.state = state["rng_state"]


class FoliageRayEvidence:
    """Sparse per-camera free/hit intervals retained outside 3D primitives."""

    def __init__(self, payload: dict | None):
        payload = payload or {}
        self.depth_coordinate_explicit = "depth_coordinate" in payload
        self.depth_coordinate = str(
            payload.get("depth_coordinate", "camera_z")
        )
        if self.depth_coordinate != "camera_z":
            raise RuntimeError(
                "Foliage ray intervals must be expressed in calibrated "
                f"camera z-depth, got {self.depth_coordinate!r}"
            )
        self.camera_ids = payload.get(
            "camera_ids", torch.empty(0, dtype=torch.int32)
        ).cpu()
        self.pixels = payload.get(
            "pixels", torch.empty(0, 2)
        ).cpu()
        self.source_image_sizes = payload.get(
            "source_image_sizes", torch.empty(0, 2, dtype=torch.int32)
        ).cpu()
        self.free_end = payload.get(
            "free_end_depth", torch.empty(0)
        ).cpu()
        self.hit_start = payload.get(
            "hit_start_depth", torch.empty(0)
        ).cpu()
        self.hit_end = payload.get(
            "hit_end_depth", torch.empty(0)
        ).cpu()
        self.observation_type = payload.get(
            "observation_type", torch.empty(0, dtype=torch.int8)
        ).cpu()
        self.confidence = payload.get(
            "confidence", torch.empty(0)
        ).cpu()
        self.offsets = payload.get(
            "offsets", torch.empty(0, dtype=torch.int64)
        ).to(dtype=torch.int64).cpu()
        count = len(self.camera_ids)
        if count and self.source_image_sizes.shape != (count, 2):
            raise RuntimeError(
                "Foliage ray evidence is missing its source-raster "
                "resolution; rebuild initialization with v18 or newer"
            )
        fields = {
            "pixels": self.pixels,
            "source_image_sizes": self.source_image_sizes,
            "free_end_depth": self.free_end,
            "hit_start_depth": self.hit_start,
            "hit_end_depth": self.hit_end,
            "observation_type": self.observation_type,
            "confidence": self.confidence,
        }
        mismatched = {
            name: len(value)
            for name, value in fields.items()
            if len(value) != count
        }
        if mismatched:
            raise RuntimeError(
                "Foliage ray evidence arrays do not share one row count: "
                f"camera_ids={count}, mismatched={mismatched}"
            )
        if count:
            if self.pixels.shape != (count, 2):
                raise RuntimeError("Foliage ray pixels must have shape [N,2]")
            if bool((self.source_image_sizes <= 0).any()):
                raise RuntimeError(
                    "Foliage ray source image sizes must be positive"
                )
            upper = self.source_image_sizes.to(self.pixels.dtype)
            # Initialization archives produced before the half-open endpoint
            # repair can contain a float64 projection that was valid before
            # serialization but rounded to exactly width/height in float32.
            # Migrate only that exact representation edge case.  Values
            # materially beyond the endpoint still fail the strict check
            # below, so this cannot hide a bad camera/raster contract.
            rounded_endpoint = (
                torch.isfinite(self.pixels).all(dim=1)
                & (self.pixels >= 0).all(dim=1)
                & (self.pixels <= upper).all(dim=1)
                & (self.pixels == upper).any(dim=1)
            )
            self.endpoint_rounding_repairs = int(
                rounded_endpoint.sum()
            )
            if self.endpoint_rounding_repairs:
                strict_upper = torch.nextafter(
                    upper,
                    torch.full_like(upper, -torch.inf),
                )
                self.pixels = torch.minimum(
                    self.pixels, strict_upper
                )
            valid_pixels = (
                torch.isfinite(self.pixels).all(dim=1)
                & (self.pixels >= 0).all(dim=1)
                & (self.pixels < upper).all(dim=1)
            )
            if not bool(valid_pixels.all()):
                raise RuntimeError(
                    "Foliage ray pixels fall outside their declared source "
                    "raster"
                )
            hit_rows = self.observation_type > 0
            finite_hit_intervals = (
                torch.isfinite(self.free_end)
                & torch.isfinite(self.hit_start)
                & torch.isfinite(self.hit_end)
            )
            valid_hit_intervals = (
                finite_hit_intervals
                & (self.free_end >= 0)
                & (self.free_end <= self.hit_start)
                & (self.hit_start < self.hit_end)
            )
            invalid_hit_rows = hit_rows & ~valid_hit_intervals
            if bool(invalid_hit_rows.any()):
                invalid_count = int(invalid_hit_rows.sum())
                overlap_count = int(
                    (
                        hit_rows
                        & torch.isfinite(self.free_end)
                        & torch.isfinite(self.hit_start)
                        & (self.free_end > self.hit_start)
                    ).sum()
                )
                raise RuntimeError(
                    "Foliage hit ray intervals violate the strict "
                    "free_end <= hit_start < hit_end contract: "
                    f"invalid_rows={invalid_count}, "
                    f"overlap_rows={overlap_count}"
                )
        else:
            self.endpoint_rounding_repairs = 0
        if self.offsets.numel():
            if self.offsets.ndim != 1 or int(self.offsets[0]) != 0:
                raise RuntimeError(
                    "Foliage ray offsets must be a one-dimensional prefix sum"
                )
            if bool((self.offsets[1:] < self.offsets[:-1]).any()):
                raise RuntimeError("Foliage ray offsets are not monotonic")
            bound_count = int(self.offsets[-1])
            if bound_count > count:
                raise RuntimeError(
                    "Foliage ray offsets exceed the ray count"
                )
            bound_ids = torch.repeat_interleave(
                torch.arange(
                    len(self.offsets) - 1, dtype=torch.int64
                ),
                self.offsets[1:] - self.offsets[:-1],
            )
            # The prefix of the table remains associated with visual-hull
            # seed identities for lineage diagnostics.  A trailing suffix is
            # allowed to contain dense observation-space rays that were
            # sampled directly from a calibrated tree component.  Those rays
            # deliberately have no seed owner: the global same-ray factor
            # must remain valid after every candidate on that ray is retired,
            # and it must be able to supervise newly born/split candidates.
            self.primitive_ids = torch.full(
                (count,), -1, dtype=torch.int64
            )
            self.primitive_ids[:bound_count] = bound_ids
        else:
            # Old states can still use the image-space diagnostic factor, but
            # they cannot claim a per-candidate posterior.
            self.primitive_ids = torch.full(
                (count,), -1, dtype=torch.int64
            )
        effective_rows = self.observation_type != 0
        self.effective_rows = effective_rows
        self.camera_id_values = torch.unique(
            self.camera_ids[effective_rows].to(dtype=torch.int64),
            sorted=True,
        )
        self.camera_rows = {}
        self.camera_hit_rows = {}
        self.camera_samplers = {}
        # Group the table once. Scanning all N rows separately for every
        # camera is O(N*C) (about 1.6 billion comparisons for Cambridge v64)
        # and dominated every restart. The composite key preserves the
        # original row order within each camera, so sampler determinism and
        # checkpoint continuation remain unchanged.
        effective_indices = torch.nonzero(
            effective_rows, as_tuple=False
        ).flatten()
        effective_camera_ids = self.camera_ids[
            effective_indices
        ].to(dtype=torch.int64)
        if len(effective_indices):
            grouping_key = (
                effective_camera_ids * (count + 1)
                + effective_indices
            )
            grouping_order = torch.argsort(grouping_key)
            grouped_rows = effective_indices[grouping_order]
            grouped_camera_ids = effective_camera_ids[grouping_order]
            grouped_values, grouped_counts = torch.unique_consecutive(
                grouped_camera_ids, return_counts=True
            )
        else:
            grouped_rows = effective_indices
            grouped_values = effective_camera_ids
            grouped_counts = torch.empty(0, dtype=torch.int64)
        offset = 0
        for camera_id, camera_count in zip(
            grouped_values.tolist(), grouped_counts.tolist()
        ):
            rows = grouped_rows[offset : offset + int(camera_count)]
            offset += int(camera_count)
            self.camera_rows[int(camera_id)] = rows
            self.camera_hit_rows[int(camera_id)] = rows[
                self.observation_type[rows] > 0
            ]
            self.camera_samplers[int(camera_id)] = EvidenceEpochSampler(
                len(rows), seed=7_919 + int(camera_id) * 104_729
            )
        # A topology mutation replaces the metadata buffer and therefore
        # changes its data pointer.  Cache the evidence-id ordering between
        # those sparse events instead of scanning every live volume on every
        # ray-factor update.
        self._descendant_cache_signature = None
        self._descendant_sorted_ids = None
        self._descendant_sorted_rows = None
        # A non-empty table is not proof that optimization ever consumes it.
        # Keep runtime counters so checkpoints/results can distinguish
        # "artifact present" from "factor exercised on the intended rays".
        self.factor_calls = 0
        self.factor_calls_with_rays = 0
        self.sampled_rays = 0
        self.sampled_confirmed_free = 0
        self.sampled_hits = 0
        self.sampled_unknown = 0
        self.consumed_camera_ids: set[int] = set()
        self.interval_factor_calls = 0
        self.interval_factor_calls_with_candidates = 0
        self.interval_candidate_evaluations = 0
        self.interval_canonical_candidate_evaluations = 0
        self.interval_exact_dynamic_candidate_evaluations = 0
        self.interval_row_visits = torch.zeros(count, dtype=torch.int64)
        # This records ownerless rays that were consumed by an *already
        # independently verified* canonical lineage. It is an audit of
        # factor routing, never a promotion mask: no ownerless observation
        # can change a primitive's role, support count or persistence.
        self.ownerless_canonical_verified = torch.zeros(
            count, dtype=torch.bool
        )
        self.sampled_ownerless_hit_rows = 0
        self.ownerless_canonical_hit_rows = 0
        self.ownerless_dynamic_hit_rows = 0
        self.ownerless_missing_dynamic_hit_rows = 0
        self.ownerless_supported_hit_rows = 0
        self.ownerless_missing_supported_hit_rows = 0
        self.verified_ownerless_hit_rows = 0
        self.excluded_ownerless_hit_rows = 0
        self._latest_uncovered_hit_proposals = None

    @torch.no_grad()
    def sample_hit_depth_posterior(
        self,
        camera_id: int,
        render_pixels: torch.Tensor,
        *,
        render_width: int,
        render_height: int,
        camera_z_depth: torch.Tensor,
        depth_tolerance: torch.Tensor,
        maximum_source_pixel_distance: float = 12.0,
        nearest_candidates: int = 8,
    ) -> dict[str, torch.Tensor]:
        """Query immutable hit intervals at current candidate projections.

        The persisted observation table is deliberately independent of the
        live Gaussian topology.  This lookup therefore answers the question
        needed after a split: does a *real calibrated ray near the child's
        current image projection* contain that child in its hit interval?
        It does not consume the epoch sampler and cannot promote ownership.

        Ray pixels are stored in their native source raster while topology
        adaptation may run at a resized training resolution.  The query is
        converted back to the declared source raster before the spatial and
        camera-z interval tests.  A small K-nearest set is considered so the
        closest image-space ray cannot hide another nearby layer whose depth
        interval actually contains the child.
        """
        render_pixels = torch.as_tensor(render_pixels)
        device = render_pixels.device
        dtype = render_pixels.dtype
        query_count = int(len(render_pixels))
        empty_bool = torch.zeros(query_count, dtype=torch.bool, device=device)
        empty_float = torch.full(
            (query_count,), float("nan"), dtype=dtype, device=device
        )
        result = {
            "spatial_candidate": empty_bool.clone(),
            "depth_consistent": empty_bool.clone(),
            "hit_start_depth": empty_float.clone(),
            "hit_end_depth": empty_float.clone(),
            "confidence": empty_float.clone(),
            "source_pixel_distance": empty_float.clone(),
        }
        if not query_count:
            return result
        rows = self.camera_hit_rows.get(int(camera_id))
        if rows is None or not len(rows):
            return result
        if int(render_width) <= 0 or int(render_height) <= 0:
            raise ValueError("Render raster dimensions must be positive")
        source_sizes = self.source_image_sizes[rows]
        source_size = source_sizes[0]
        if not bool((source_sizes == source_size).all()):
            raise RuntimeError(
                "One foliage-ray camera contains multiple source rasters"
            )
        source_width = float(source_size[0])
        source_height = float(source_size[1])
        query_source = render_pixels.to(dtype=dtype).clone()
        query_source[:, 0] *= source_width / float(render_width)
        query_source[:, 1] *= source_height / float(render_height)
        evidence_pixels = self.pixels[rows].to(device=device, dtype=dtype)
        hit_start = self.hit_start[rows].to(device=device, dtype=dtype)
        hit_end = self.hit_end[rows].to(device=device, dtype=dtype)
        confidence = self.confidence[rows].to(device=device, dtype=dtype)
        camera_z_depth = torch.as_tensor(
            camera_z_depth, device=device, dtype=dtype
        ).reshape(-1)
        depth_tolerance = torch.as_tensor(
            depth_tolerance, device=device, dtype=dtype
        ).reshape(-1)
        if len(camera_z_depth) != query_count or len(depth_tolerance) != query_count:
            raise ValueError("Ray-posterior query tensors must share one row count")
        k = min(max(int(nearest_candidates), 1), int(len(rows)))
        maximum_distance = float(maximum_source_pixel_distance)
        # Bound the temporary QxN distance matrix.  At Cambridge scale each
        # camera has roughly 3--6k hit rays, so 1,024 queries stay comfortably
        # below the memory of one renderer frame while avoiding Python
        # per-candidate searches.
        for begin in range(0, query_count, 1024):
            end = min(begin + 1024, query_count)
            distances = torch.cdist(
                query_source[begin:end], evidence_pixels
            )
            nearest_distance, nearest_index = torch.topk(
                distances, k, dim=1, largest=False, sorted=True
            )
            spatial = nearest_distance <= maximum_distance
            local_depth = camera_z_depth[begin:end, None]
            local_tolerance = depth_tolerance[begin:end, None]
            local_start = hit_start[nearest_index]
            local_end = hit_end[nearest_index]
            depth_ok = (
                torch.isfinite(local_depth)
                & (local_depth > 0.05)
                & (local_depth >= local_start - local_tolerance)
                & (local_depth <= local_end + local_tolerance)
            )
            accepted = spatial & depth_ok
            result["spatial_candidate"][begin:end] = spatial.any(dim=1)
            accepted_any = accepted.any(dim=1)
            result["depth_consistent"][begin:end] = accepted_any
            # Invalid candidates receive +inf so argmin selects the nearest
            # depth-consistent observation rather than merely the nearest ray.
            selected_slot = torch.where(
                accepted,
                nearest_distance,
                nearest_distance.new_full(nearest_distance.shape, float("inf")),
            ).argmin(dim=1)
            selected_row = nearest_index.gather(
                1, selected_slot[:, None]
            )[:, 0]
            local_rows = torch.arange(end - begin, device=device)
            selected_distance = nearest_distance[
                local_rows, selected_slot
            ]
            for name, values in (
                ("hit_start_depth", hit_start),
                ("hit_end_depth", hit_end),
                ("confidence", confidence),
            ):
                selected = values[selected_row]
                result[name][begin:end] = torch.where(
                    accepted_any, selected, result[name][begin:end]
                )
            result["source_pixel_distance"][begin:end] = torch.where(
                accepted_any,
                selected_distance,
                result["source_pixel_distance"][begin:end],
            )
        return result

    def pop_uncovered_hit_proposals(self) -> dict | None:
        """Return and clear static birth proposals from the latest ray batch."""
        proposals = self._latest_uncovered_hit_proposals
        self._latest_uncovered_hit_proposals = None
        return proposals

    def audit(self) -> dict:
        kind = self.observation_type
        source_camera_count = int(torch.unique(self.camera_ids).numel())
        effective_visits = self.interval_row_visits[self.effective_rows]
        visited_effective = int((effective_visits > 0).sum())
        camera_epoch_audits = {
            str(camera_id): sampler.audit()
            for camera_id, sampler in self.camera_samplers.items()
        }
        completed_epochs = [
            audit["completed_epochs"]
            for audit in camera_epoch_audits.values()
        ]
        return {
            "ray_count": int(len(kind)),
            "hit_count": int((kind > 0).sum()),
            "confirmed_free_count": int((kind < 0).sum()),
            "unknown_count": int((kind == 0).sum()),
            "camera_count": source_camera_count,
            "factor_camera_count": int(len(self.camera_id_values)),
            "explicit_source_resolution": True,
            "endpoint_rounding_repairs": int(
                self.endpoint_rounding_repairs
            ),
            "primitive_association_retained": bool(
                (self.primitive_ids >= 0).any()
            ),
            "seed_bound_ray_count": int(
                (self.primitive_ids >= 0).sum()
            ),
            "observation_space_ray_count": int(
                (self.primitive_ids < 0).sum()
            ),
            "posterior_factor": "global_same_ray_analytic_transmittance",
            "stored_depth_coordinate": self.depth_coordinate,
            "analytic_depth_coordinate": "unit_ray_distance",
            "depth_coordinate_conversion": (
                "t=z/direction_camera_z"
            ),
            "depth_coordinate_explicit": self.depth_coordinate_explicit,
            "strict_hit_interval_validation": True,
            "unknown_is_excluded_from_free_loss": True,
            "confidence_normalization": (
                "depth_geometry_weighted_sum_over_ray_count__missing_"
                "confidence_mass_routes_to_opacity_only_existence"
            ),
            "factor_calls": int(self.factor_calls),
            "factor_calls_with_rays": int(self.factor_calls_with_rays),
            "sampled_rays": int(self.sampled_rays),
            "sampled_confirmed_free": int(
                self.sampled_confirmed_free
            ),
            "sampled_hits": int(self.sampled_hits),
            "sampled_unknown": int(self.sampled_unknown),
            "consumed_camera_count": len(self.consumed_camera_ids),
            "consumed_camera_coverage": (
                len(self.consumed_camera_ids)
                / max(len(self.camera_id_values), 1)
            ),
            "interval_factor_calls": int(self.interval_factor_calls),
            "interval_factor_calls_with_candidates": int(
                self.interval_factor_calls_with_candidates
            ),
            "interval_candidate_evaluations": int(
                self.interval_candidate_evaluations
            ),
            "interval_canonical_candidate_evaluations": int(
                self.interval_canonical_candidate_evaluations
            ),
            "interval_exact_dynamic_candidate_evaluations": int(
                self.interval_exact_dynamic_candidate_evaluations
            ),
            "interval_effective_rows": int(self.effective_rows.sum()),
            "interval_unique_rows": visited_effective,
            "interval_row_coverage": float(
                visited_effective / max(len(effective_visits), 1)
            )
            if len(effective_visits)
            else 0.0,
            "interval_minimum_visits": (
                int(effective_visits.min())
                if len(effective_visits)
                else 0
            ),
            "interval_maximum_visits": (
                int(effective_visits.max())
                if len(effective_visits)
                else 0
            ),
            "camera_epoch_minimum": (
                int(min(completed_epochs)) if completed_epochs else 0
            ),
            "camera_epoch_maximum": (
                int(max(completed_epochs)) if completed_epochs else 0
            ),
            "camera_epoch_audits": camera_epoch_audits,
            "camera_schedule_contract": (
                "minimum_normalized_row_visits_first__stable_camera_id_tie"
            ),
            "ownerless_canonical_contract": (
                "ownerless_never_promotes__existing_cross_sequence_"
                "canonical_lineage_plus_exact_camera_dynamic_residual"
            ),
            "sampled_ownerless_hit_rows": int(
                self.sampled_ownerless_hit_rows
            ),
            "ownerless_canonical_factor_rows": int(
                self.ownerless_canonical_hit_rows
            ),
            "ownerless_dynamic_factor_rows": int(
                self.ownerless_dynamic_hit_rows
            ),
            "ownerless_missing_dynamic_hit_rows": int(
                self.ownerless_missing_dynamic_hit_rows
            ),
            "ownerless_supported_hit_rows": int(
                self.ownerless_supported_hit_rows
            ),
            "ownerless_missing_supported_hit_rows": int(
                self.ownerless_missing_supported_hit_rows
            ),
            "verified_ownerless_hit_rows": int(
                self.verified_ownerless_hit_rows
            ),
            "excluded_ownerless_hit_rows": int(
                self.excluded_ownerless_hit_rows
            ),
            "ownerless_excluded_from_canonical_promotion_rows": int(
                self.excluded_ownerless_hit_rows
            ),
            "unique_verified_ownerless_rows": int(
                self.ownerless_canonical_verified.sum()
            ),
        }

    def capture_runtime_state(self) -> dict:
        """Persist posterior coverage and ownerless factor routing."""
        return {
            "version": "foliage-ray-runtime-v5-shared-canonical-residual",
            "row_count": int(len(self.camera_ids)),
            "factor_calls": int(self.factor_calls),
            "factor_calls_with_rays": int(
                self.factor_calls_with_rays
            ),
            "sampled_rays": int(self.sampled_rays),
            "sampled_confirmed_free": int(
                self.sampled_confirmed_free
            ),
            "sampled_hits": int(self.sampled_hits),
            "sampled_unknown": int(self.sampled_unknown),
            "consumed_camera_ids": sorted(self.consumed_camera_ids),
            "interval_factor_calls": int(self.interval_factor_calls),
            "interval_factor_calls_with_candidates": int(
                self.interval_factor_calls_with_candidates
            ),
            "interval_candidate_evaluations": int(
                self.interval_candidate_evaluations
            ),
            "interval_canonical_candidate_evaluations": int(
                self.interval_canonical_candidate_evaluations
            ),
            "interval_exact_dynamic_candidate_evaluations": int(
                self.interval_exact_dynamic_candidate_evaluations
            ),
            "interval_row_visits": self.interval_row_visits.clone(),
            "camera_sampler_states": {
                str(camera_id): sampler.capture_state()
                for camera_id, sampler in self.camera_samplers.items()
            },
            "ownerless_canonical_verified": (
                self.ownerless_canonical_verified.clone()
            ),
            "sampled_ownerless_hit_rows": int(
                self.sampled_ownerless_hit_rows
            ),
            "ownerless_canonical_hit_rows": int(
                self.ownerless_canonical_hit_rows
            ),
            "ownerless_dynamic_hit_rows": int(
                self.ownerless_dynamic_hit_rows
            ),
            "ownerless_missing_dynamic_hit_rows": int(
                self.ownerless_missing_dynamic_hit_rows
            ),
            "ownerless_supported_hit_rows": int(
                self.ownerless_supported_hit_rows
            ),
            "ownerless_missing_supported_hit_rows": int(
                self.ownerless_missing_supported_hit_rows
            ),
            "verified_ownerless_hit_rows": int(
                self.verified_ownerless_hit_rows
            ),
            "excluded_ownerless_hit_rows": int(
                self.excluded_ownerless_hit_rows
            ),
        }

    def restore_runtime_state(self, state: dict | None) -> None:
        """Restore an exact posterior-factor continuation contract."""
        if not state:
            return
        version = state.get("version")
        if version not in {
            "foliage-ray-runtime-v2-independent-canonical",
            "foliage-ray-runtime-v3-camera-epochs",
            "foliage-ray-runtime-v4-owner-audit",
            "foliage-ray-runtime-v5-shared-canonical-residual",
        }:
            raise RuntimeError(
                "Unsupported foliage ray runtime state version: "
                f"{version!r}"
            )
        row_count = int(state.get("row_count", -1))
        if row_count != len(self.camera_ids):
            raise RuntimeError(
                "Foliage ray runtime row count changed across resume: "
                f"{row_count} != {len(self.camera_ids)}"
            )
        row_visits = torch.as_tensor(
            state["interval_row_visits"], dtype=torch.int64
        ).cpu()
        verified = torch.as_tensor(
            state["ownerless_canonical_verified"],
            dtype=torch.bool,
        ).cpu()
        expected_shape = (len(self.camera_ids),)
        if (
            row_visits.shape != expected_shape
            or verified.shape != expected_shape
        ):
            raise RuntimeError(
                "Foliage ray runtime masks do not match the evidence table"
            )
        if (
            version != "foliage-ray-runtime-v5-shared-canonical-residual"
            and bool(verified.any())
        ):
            raise RuntimeError(
                "A legacy ownerless-ray runtime state unexpectedly contains "
                "canonical-consumption rows"
            )
        integer_fields = (
            "factor_calls",
            "factor_calls_with_rays",
            "sampled_rays",
            "sampled_confirmed_free",
            "sampled_hits",
            "sampled_unknown",
            "interval_factor_calls",
            "interval_factor_calls_with_candidates",
            "interval_candidate_evaluations",
            "interval_canonical_candidate_evaluations",
            "interval_exact_dynamic_candidate_evaluations",
            "sampled_ownerless_hit_rows",
            "ownerless_canonical_hit_rows",
            "ownerless_dynamic_hit_rows",
            "ownerless_missing_dynamic_hit_rows",
            "ownerless_supported_hit_rows",
            "ownerless_missing_supported_hit_rows",
            "verified_ownerless_hit_rows",
            "excluded_ownerless_hit_rows",
        )
        for field in integer_fields:
            value = int(state.get(field, 0))
            if value < 0:
                raise RuntimeError(
                    f"Negative foliage ray runtime counter {field}"
                )
            setattr(self, field, value)
        consumed = {
            int(value) for value in state.get("consumed_camera_ids", ())
        }
        known = {int(value) for value in self.camera_id_values.tolist()}
        if not consumed.issubset(known):
            raise RuntimeError(
                "Foliage ray runtime references cameras outside its table"
            )
        self.consumed_camera_ids = consumed
        self.interval_row_visits.copy_(row_visits)
        self.ownerless_canonical_verified.copy_(verified)
        if version in {
            "foliage-ray-runtime-v3-camera-epochs",
            "foliage-ray-runtime-v4-owner-audit",
            "foliage-ray-runtime-v5-shared-canonical-residual",
        }:
            sampler_states = state.get("camera_sampler_states", {})
            expected = {
                str(camera_id) for camera_id in self.camera_samplers
            }
            if set(sampler_states) != expected:
                raise RuntimeError(
                    "Foliage ray camera sampler identities changed across "
                    "resume"
                )
            for camera_id, sampler in self.camera_samplers.items():
                sampler.restore_state(sampler_states[str(camera_id)])
        else:
            # v2 selected contiguous blocks from one deterministic shuffled
            # order but persisted only a boolean coverage mask.  Recover the
            # longest visited prefix when possible and otherwise put all
            # never-visited rows first.  This migration cannot reconstruct
            # repeated visits, but it never throws away known coverage or
            # restarts at the same already-visited prefix.
            for camera_id, sampler in self.camera_samplers.items():
                rows = self.camera_rows[camera_id]
                local_visited = (
                    self.interval_row_visits[rows].numpy() > 0
                )
                ordered_visited = local_visited[sampler.order]
                prefix = 0
                while (
                    prefix < sampler.count
                    and bool(ordered_visited[prefix])
                ):
                    prefix += 1
                sampler.visits = local_visited.astype(
                    np.int64, copy=True
                )
                if prefix == sampler.count and sampler.count:
                    sampler.epoch = 1
                    sampler.order = sampler.rng.permutation(
                        sampler.count
                    )
                    sampler.cursor = 0
                elif not bool(ordered_visited[prefix:].any()):
                    sampler.cursor = prefix
                else:
                    sampler.order = np.concatenate(
                        [
                            sampler.order[ordered_visited],
                            sampler.order[~ordered_visited],
                        ]
                    )
                    sampler.cursor = int(ordered_visited.sum())

    def scheduled_camera_id(self, update: int) -> int | None:
        if not len(self.camera_id_values):
            return None
        # Camera tables differ by almost two orders of magnitude in Cambridge.
        # A uniform camera round-robin therefore repeated small tables while
        # large tree views remained mostly untouched.  Schedule the camera
        # with the least normalized row visitation instead.  The per-camera
        # sampler still owns permutation/cursor/epoch state; this outer policy
        # only decides which independent epoch advances next.
        del update
        best_camera = None
        best_progress = None
        for camera_id in self.camera_id_values.tolist():
            sampler = self.camera_samplers[int(camera_id)]
            progress = (
                float(sampler.epoch)
                + float(sampler.cursor) / max(sampler.count, 1)
            )
            if (
                best_progress is None
                or progress < best_progress - 1e-12
            ):
                best_camera = int(camera_id)
                best_progress = progress
        return best_camera

    def _next_camera_rows(
        self, camera_id: int, maximum_rays: int
    ) -> torch.Tensor:
        rows = self.camera_rows.get(int(camera_id))
        sampler = self.camera_samplers.get(int(camera_id))
        if rows is None or sampler is None:
            return torch.empty(0, dtype=torch.int64)
        local_rows = sampler.next(int(maximum_rays))
        if not len(local_rows):
            return rows[:0]
        selected = rows[torch.from_numpy(local_rows)]
        self.interval_row_visits[selected] += 1
        return selected

    def _descendants_for_sources(self, foliage, source_ids):
        # A split child cannot claim its parent's evidence row as already
        # verified at the displaced centre.  It may nevertheless use that row
        # as a candidate for the differentiable ray-interval recheck.
        verified_ids = foliage.evidence_primitive_id
        candidate_ids = getattr(
            foliage, "candidate_evidence_primitive_id", verified_ids
        )
        current_ids = torch.where(
            verified_ids >= 0, verified_ids, candidate_ids
        )
        signature = (
            str(current_ids.device),
            int(verified_ids.data_ptr()),
            int(candidate_ids.data_ptr()),
            int(current_ids.numel()),
        )
        if signature != self._descendant_cache_signature:
            valid_rows = torch.nonzero(
                current_ids >= 0, as_tuple=False
            ).flatten()
            if len(valid_rows):
                order = torch.argsort(current_ids[valid_rows])
                self._descendant_sorted_rows = valid_rows[order]
                self._descendant_sorted_ids = current_ids[
                    self._descendant_sorted_rows
                ]
            else:
                self._descendant_sorted_rows = valid_rows
                self._descendant_sorted_ids = current_ids[:0]
            self._descendant_cache_signature = signature
        sorted_ids = self._descendant_sorted_ids
        sorted_rows = self._descendant_sorted_rows
        if sorted_ids is None or not len(sorted_ids):
            empty = torch.empty(
                0, dtype=torch.int64, device=current_ids.device
            )
            return empty, empty
        starts = torch.searchsorted(sorted_ids, source_ids, right=False)
        ends = torch.searchsorted(sorted_ids, source_ids, right=True)
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            empty = torch.empty(
                0, dtype=torch.int64, device=current_ids.device
            )
            return empty, empty
        record = torch.repeat_interleave(
            torch.arange(
                len(source_ids),
                dtype=torch.int64,
                device=current_ids.device,
            ),
            lengths,
        )
        prefix = torch.cumsum(lengths, dim=0)
        relative = torch.arange(
            total, dtype=torch.int64, device=current_ids.device
        ) - torch.repeat_interleave(prefix - lengths, lengths)
        positions = torch.repeat_interleave(starts, lengths) + relative
        return sorted_rows[positions], record

    def interval_factor(
        self,
        view,
        foliage,
        *,
        maximum_rays: int = 256,
        maximum_candidates_per_ray: int = 96,
        sample_update: int = 0,
        canonical_candidate_mask: torch.Tensor | None = None,
        canonical_hit_candidate_mask: torch.Tensor | None = None,
        hit_opacity_gradient_scale: float = 1.0,
        return_loss_components: bool = False,
    ):
        """Evaluate all nearby canonical Gaussians on each measured ray.

        A retained seed id says which visual-hull proposal produced a ray; it
        does *not* say that only descendants of that proposal may explain the
        observation.  Treating every seed independently made overlapping
        proposals each maximize their own hit opacity, recreating the solid
        translucent crown that the ray posterior was meant to prevent.

        This factor first selects nearby canonical volumes in screen/ray
        space, then evaluates their anisotropic Gaussian density analytically.
        Optical depth is accumulated over the complete local ray, irrespective
        of lineage.  A hit therefore has the proper transmittance likelihood

            tau_before - log(1 - exp(-tau_inside)),

        while a confirmed-free observation penalizes only ``tau_before``.
        ``canonical_candidate_mask`` grants globally valid free-space
        participation. ``canonical_hit_candidate_mask`` is the narrower
        positive-hit permission; when omitted it defaults to the former for
        exact backwards compatibility. Cross-sequence seed-bound hits use
        canonical mass. Dense observation-space hits use the sum of (a)
        canonical lineages that were independently established before this
        observation, (b) sequence-local residual leaves whose immutable
        support table names the exact scheduled camera, and (c) fused static
        detail whose cross-sequence consensus table names that camera.  The
        last route preserves exact ray ownership when static deployment
        removes the dynamic row. The dense ray can optimize an existing
        verified lineage, but can never promote a new one or alter its
        support metadata. Confirmed-free rays constrain every static branch
        visible in that camera.  ``return_loss_components`` exposes the exact
        free/pre-hit and hit/existence scalars used to build the returned
        total.  They share one forward evaluation but remain separate autograd
        sources, so a row constrained by both observations cannot hide a
        contradiction through a net gradient before the ownership policy sees
        it.
        """
        hit_opacity_gradient_scale = float(hit_opacity_gradient_scale)
        if not 0.0 <= hit_opacity_gradient_scale <= 1.0:
            raise ValueError("hit_opacity_gradient_scale must lie in [0,1]")
        self.interval_factor_calls += 1
        self._latest_uncovered_hit_proposals = None
        zero = foliage.xyz.new_zeros(())
        selected = self._next_camera_rows(
            int(view.colmap_id), int(maximum_rays)
        )
        if not len(selected):
            audit = {
                "rays": 0,
                "candidate_evaluations": 0,
                "canonical_candidate_evaluations": 0,
                "canonical_hit_candidate_evaluations": 0,
                "exact_dynamic_candidate_evaluations": 0,
                "exact_static_candidate_evaluations": 0,
                "confirmed_free_rays": 0,
                "hit_rays": 0,
                "unknown_rays": 0,
                "free": 0.0,
                "hit": 0.0,
                "behind_mass": 0.0,
                "hit_opacity_gradient_scale": hit_opacity_gradient_scale,
            }
            if return_loss_components:
                return zero, audit, {
                    "free": zero,
                    "confirmed_free": zero,
                    "hit_prehit": zero,
                    "hit": zero,
                }
            return zero, audit
        device, dtype = foliage.xyz.device, foliage.xyz.dtype
        pixels = self.pixels[selected].to(device=device, dtype=dtype)
        source_size = self.source_image_sizes[selected].to(
            device=device, dtype=dtype
        )
        render_x = pixels[:, 0] * (
            float(view.image_width) / source_size[:, 0]
        )
        render_y = pixels[:, 1] * (
            float(view.image_height) / source_size[:, 1]
        )
        direction_camera = torch.stack(
            [
                (render_x - float(view.cx)) / float(view.focal_x),
                (render_y - float(view.cy)) / float(view.focal_y),
                torch.ones_like(render_x),
            ],
            dim=-1,
        )
        direction_camera = F.normalize(direction_camera, dim=-1)
        # The persisted posterior intervals use the same camera-z convention
        # as MASt3R pointmaps and the mixed renderer depth output.  The
        # analytic Gaussian integral below, however, parameterizes a *unit*
        # ray by Euclidean distance.  Comparing those two coordinates
        # directly shifts an off-axis hit behind its measured interval by a
        # factor of 1 / ray_z (about 1.04--1.15 in Cambridge).  Convert the
        # immutable z-depth observations to unit-ray distance exactly once.
        # This is not an empirical scale correction: for
        # X_cam = t * direction_camera, z = t * direction_camera.z.
        z_to_ray_distance = direction_camera[:, 2].clamp_min(1e-6).reciprocal()
        world_from_camera = torch.as_tensor(
            view.R, device=device, dtype=dtype
        )
        direction_world = F.normalize(
            direction_camera @ world_from_camera.T, dim=-1
        )
        origin = view.camera_center.to(device=device, dtype=dtype)
        free_end = (
            self.free_end[selected].to(device=device, dtype=dtype)
            * z_to_ray_distance
        )
        hit_start = self.hit_start[selected].to(
            device=device, dtype=dtype
        ) * z_to_ray_distance
        hit_end = (
            self.hit_end[selected].to(device=device, dtype=dtype)
            * z_to_ray_distance
        )
        sqrt_two = float(np.sqrt(2.0))

        def normal_cdf(value):
            return 0.5 * (
                1.0 + torch.erf(value / sqrt_two)
            )

        dynamic_mask = foliage.dynamic_leaf_mask
        if canonical_candidate_mask is None:
            canonical_candidate_mask = torch.ones_like(
                dynamic_mask, dtype=torch.bool
            )
        else:
            canonical_candidate_mask = torch.as_tensor(
                canonical_candidate_mask,
                device=device,
                dtype=torch.bool,
            ).reshape(-1)
            if len(canonical_candidate_mask) != len(foliage):
                raise ValueError(
                    "canonical candidate mask must align with foliage"
                )
        if canonical_hit_candidate_mask is None:
            canonical_hit_candidate_mask = canonical_candidate_mask
        else:
            canonical_hit_candidate_mask = torch.as_tensor(
                canonical_hit_candidate_mask,
                device=device,
                dtype=torch.bool,
            ).reshape(-1)
            if len(canonical_hit_candidate_mask) != len(foliage):
                raise ValueError(
                    "canonical hit candidate mask must align with foliage"
                )
            if bool(
                (
                    canonical_hit_candidate_mask
                    & ~canonical_candidate_mask
                ).any()
            ):
                raise ValueError(
                    "canonical hit candidates must be a subset of free-space "
                    "candidates"
                )
        support_camera_ids = foliage.support_camera_ids
        if support_camera_ids.ndim != 2 or len(support_camera_ids) != len(
            foliage
        ):
            raise RuntimeError(
                "Foliage support-camera metadata no longer aligns with the "
                "volume model"
            )
        # Cross-sequence seed-bound rays supervise the canonical crown.  A
        # dense observation-space ray instead owns the exact sequence-local
        # birth created from this calibrated camera. Static deployment turns
        # independently confirmed dynamic rows into unconditional detail but
        # preserves their contributing-camera table. That table remains
        # positive-ray ownership; neighbouring-frame rendering fallback is
        # not ray-posterior evidence.
        exact_dynamic = dynamic_mask & (
            support_camera_ids == int(view.colmap_id)
        ).any(dim=1)
        static_detail_mask = getattr(
            foliage,
            "static_leaf_mask",
            torch.zeros_like(dynamic_mask),
        )
        exact_static = static_detail_mask & (
            support_camera_ids == int(view.colmap_id)
        ).any(dim=1) & canonical_candidate_mask
        verification_state = getattr(foliage, "verification_state", None)
        verified_camera_count = getattr(
            foliage, "verified_camera_count", None
        )
        verified_sequence_count = getattr(
            foliage, "verified_sequence_count", None
        )
        if (
            verification_state is None
            or verified_camera_count is None
            or verified_sequence_count is None
        ):
            persistent_verified = torch.zeros_like(dynamic_mask)
        else:
            persistent_verified = (
                verification_state == 1
            ) & torch.where(
                static_detail_mask,
                (verified_camera_count >= 2)
                & (verified_sequence_count >= 1),
                foliage.support_sequence_count >= 2,
            )
        verified_canonical = (
            ~dynamic_mask
            & ~exact_static
            & canonical_hit_candidate_mask
            & persistent_verified
        )
        candidate_pool = torch.nonzero(
            ((~dynamic_mask) & canonical_candidate_mask)
            | exact_dynamic
            | exact_static,
            as_tuple=False,
        ).flatten()
        candidate_evaluations = 0
        canonical_candidate_evaluations = 0
        canonical_hit_candidate_evaluations = 0
        exact_dynamic_candidate_evaluations = 0
        exact_static_candidate_evaluations = 0
        rays_with_candidates = 0
        canonical_free_tau_chunks = []
        canonical_hit_tau_chunks = []
        canonical_hit_optical_tau_chunks = []
        canonical_behind_tau_chunks = []
        verified_canonical_free_tau_chunks = []
        verified_canonical_hit_tau_chunks = []
        verified_canonical_hit_optical_tau_chunks = []
        verified_canonical_behind_tau_chunks = []
        dynamic_free_tau_chunks = []
        dynamic_hit_tau_chunks = []
        dynamic_hit_optical_tau_chunks = []
        dynamic_behind_tau_chunks = []
        static_hit_tau_chunks = []
        static_hit_optical_tau_chunks = []
        static_behind_tau_chunks = []
        observation_kind = self.observation_type[selected].to(device=device)
        split_candidate_permissions = bool(
            (
                canonical_candidate_mask
                & ~canonical_hit_candidate_mask
                & ~dynamic_mask
            ).any()
        )
        # Candidate lookup is intentionally detached.  It is a sparse
        # acceleration structure, not part of the objective; the selected
        # Gaussian parameters below remain fully differentiable.
        # Bound the dense ray-by-candidate scratch matrix, not merely the ray
        # count.  At v99/20k the canonical pool had about 900k rows, so the
        # historical fixed chunk of 64 produced several simultaneous 208 MiB
        # matrices and OOMed even though only 96 candidates per ray survive.
        # Chunking rays is mathematically exact because top-k still sees the
        # complete candidate pool for every ray.
        maximum_lookup_matrix_elements = 4_000_000
        lookup_chunk = min(
            64,
            max(
                1,
                maximum_lookup_matrix_elements
                // max(int(len(candidate_pool)), 1),
            ),
        )
        if len(candidate_pool):
            with torch.no_grad():
                lookup_displacement = (
                    foliage.xyz[candidate_pool] - origin
                )
                lookup_norm2 = lookup_displacement.square().sum(-1)
                lookup_scale2 = (
                    foliage.scales[candidate_pool]
                    .amax(dim=-1)
                    .clamp_min(1e-4)
                    .square()
                )
                hit_pool_local = torch.nonzero(
                    exact_dynamic[candidate_pool]
                    | exact_static[candidate_pool]
                    | canonical_hit_candidate_mask[candidate_pool],
                    as_tuple=False,
                ).flatten()
        for begin in range(0, len(selected), lookup_chunk):
            end = min(begin + lookup_chunk, len(selected))
            ray_count = end - begin
            if not len(candidate_pool):
                empty = torch.zeros(
                    ray_count, device=device, dtype=dtype
                )
                canonical_free_tau_chunks.append(empty)
                canonical_hit_tau_chunks.append(empty)
                canonical_hit_optical_tau_chunks.append(empty)
                canonical_behind_tau_chunks.append(empty)
                verified_canonical_free_tau_chunks.append(empty)
                verified_canonical_hit_tau_chunks.append(empty)
                verified_canonical_hit_optical_tau_chunks.append(empty)
                verified_canonical_behind_tau_chunks.append(empty)
                dynamic_free_tau_chunks.append(empty)
                dynamic_hit_tau_chunks.append(empty)
                dynamic_hit_optical_tau_chunks.append(empty)
                dynamic_behind_tau_chunks.append(empty)
                static_hit_tau_chunks.append(empty)
                static_hit_optical_tau_chunks.append(empty)
                static_behind_tau_chunks.append(empty)
                continue
            with torch.no_grad():
                ray_direction = direction_world[begin:end]
                along = ray_direction @ lookup_displacement.T
                radial2 = (
                    lookup_norm2[None] - along.square()
                ).clamp_min(0)
                lookup_score = radial2 / lookup_scale2[None]
                # Candidate lookup is detached, so masking in place is safe
                # and avoids a second full ray-by-pool output plus the
                # full_like input that caused the observed 208 MiB request.
                lookup_score.masked_fill_(along <= 0, float("inf"))
                del along, radial2
                candidate_limit = min(
                    max(int(maximum_candidates_per_ray), 1),
                    len(candidate_pool),
                )
                score, local_rows = torch.topk(
                    lookup_score,
                    candidate_limit,
                    dim=1,
                    largest=False,
                    sorted=False,
                )
                # Six standard deviations contains effectively all Gaussian
                # mass while preventing a far-away top-k fallback from
                # manufacturing a candidate for an unsupported ray.
                supported = torch.isfinite(score) & (score <= 36.0)
                local_ray, slot = torch.nonzero(
                    supported, as_tuple=True
                )
                selected_pool_rows = local_rows[local_ray, slot]
                # A global free-space pool can be much denser than the
                # canonical-support positive-hit pool.  Reserve an
                # independent top-k for the latter on hit rays, then union
                # both sets. Otherwise nearby non-owned leaves could consume
                # the whole lookup budget and silently erase the very hit
                # permission this API is meant to preserve.
                if (
                    split_candidate_permissions
                    and len(hit_pool_local)
                    and bool((observation_kind[begin:end] > 0).any())
                ):
                    hit_limit = min(
                        max(int(maximum_candidates_per_ray), 1),
                        len(hit_pool_local),
                    )
                    hit_score, hit_local_rows = torch.topk(
                        lookup_score[:, hit_pool_local],
                        hit_limit,
                        dim=1,
                        largest=False,
                        sorted=False,
                    )
                    hit_supported = (
                        torch.isfinite(hit_score)
                        & (hit_score <= 36.0)
                        & (observation_kind[begin:end] > 0)[:, None]
                    )
                    hit_local_ray, hit_slot = torch.nonzero(
                        hit_supported, as_tuple=True
                    )
                    hit_selected_pool_rows = hit_pool_local[
                        hit_local_rows[hit_local_ray, hit_slot]
                    ]
                    pool_count = int(len(candidate_pool))
                    encoded = torch.cat(
                        [
                            local_ray * pool_count + selected_pool_rows,
                            hit_local_ray * pool_count
                            + hit_selected_pool_rows,
                        ]
                    ).unique()
                    local_ray = torch.div(
                        encoded, pool_count, rounding_mode="floor"
                    )
                    selected_pool_rows = encoded.remainder(pool_count)
                candidate_rows = candidate_pool[selected_pool_rows]
            if not len(candidate_rows):
                empty = torch.zeros(
                    ray_count, device=device, dtype=dtype
                )
                canonical_free_tau_chunks.append(empty)
                canonical_hit_tau_chunks.append(empty)
                canonical_hit_optical_tau_chunks.append(empty)
                canonical_behind_tau_chunks.append(empty)
                verified_canonical_free_tau_chunks.append(empty)
                verified_canonical_hit_tau_chunks.append(empty)
                verified_canonical_hit_optical_tau_chunks.append(empty)
                verified_canonical_behind_tau_chunks.append(empty)
                dynamic_free_tau_chunks.append(empty)
                dynamic_hit_tau_chunks.append(empty)
                dynamic_hit_optical_tau_chunks.append(empty)
                dynamic_behind_tau_chunks.append(empty)
                static_hit_tau_chunks.append(empty)
                static_hit_optical_tau_chunks.append(empty)
                static_behind_tau_chunks.append(empty)
                continue
            candidate_evaluations += int(len(candidate_rows))
            candidate_dynamic = dynamic_mask[candidate_rows]
            candidate_static = exact_static[candidate_rows]
            exact_dynamic_candidate_evaluations += int(
                candidate_dynamic.sum()
            )
            exact_static_candidate_evaluations += int(
                candidate_static.sum()
            )
            canonical_candidate_evaluations += int(
                (~candidate_dynamic).sum()
            )
            canonical_hit_candidate_evaluations += int(
                (
                    (~candidate_dynamic)
                    & canonical_hit_candidate_mask[candidate_rows]
                ).sum()
            )
            rays_with_candidates += int(torch.unique(local_ray).numel())
            direction = direction_world[begin:end][local_ray]
            displacement = foliage.xyz[candidate_rows] - origin
            quaternion = foliage.normalized_quaternions[candidate_rows]
            w, x, y, z = quaternion.unbind(-1)
            rotation = torch.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ],
                dim=-1,
            ).reshape(-1, 3, 3)
            world_to_local = rotation.transpose(1, 2)
            local_direction = torch.bmm(
                world_to_local, direction[:, :, None]
            ).squeeze(-1)
            local_displacement = torch.bmm(
                world_to_local, displacement[:, :, None]
            ).squeeze(-1)
            inverse_variance = foliage.scales[
                candidate_rows
            ].clamp_min(1e-4).square().reciprocal()
            ray_precision = (
                local_direction.square() * inverse_variance
            ).sum(-1).clamp_min(1e-8)
            closest_depth = (
                local_direction
                * local_displacement
                * inverse_variance
            ).sum(-1) / ray_precision
            closest = (
                local_displacement
                - closest_depth[:, None] * local_direction
            )
            radial_mahalanobis = (
                closest.square() * inverse_variance
            ).sum(-1).clamp_min(0)
            local_alpha = (
                foliage.opacities[candidate_rows]
                * torch.exp(
                    -0.5 * radial_mahalanobis.clamp_max(80)
                )
            ).clamp(0, 1 - 1e-6)
            local_tau = -torch.log1p(-local_alpha)
            # Hit geometry remains fully differentiable in centre, scale and
            # rotation, while late static phases may anneal only the opacity
            # derivative of the unbounded existence likelihood.  Free-space
            # and behind-interval mass keep the unmodified tau above, so this
            # cannot hide a contradiction or weaken optical cleanup.
            candidate_opacity = foliage.opacities[candidate_rows]
            hit_opacity = candidate_opacity.detach() + (
                hit_opacity_gradient_scale
                * (candidate_opacity - candidate_opacity.detach())
            )
            local_alpha_hit = (
                hit_opacity
                * torch.exp(
                    -0.5 * radial_mahalanobis.clamp_max(80)
                )
            ).clamp(0, 1 - 1e-6)
            local_tau_hit = -torch.log1p(-local_alpha_hit)
            # A tree-labelled owner pixel is strong optical-existence
            # evidence even when its monocular metric depth is uncertain.
            # Build a second tau whose footprint/depth terms are constants:
            # it can update opacity, but cannot drag centres/scales toward a
            # broad interval.  The confidence-weighted analytic tau below
            # remains the geometry posterior.
            local_alpha_optical = (
                hit_opacity
                * torch.exp(
                    -0.5 * radial_mahalanobis.detach().clamp_max(80)
                )
            ).clamp(0, 1 - 1e-6)
            local_tau_optical = -torch.log1p(-local_alpha_optical)
            longitudinal_sigma = ray_precision.rsqrt().clamp_min(1e-4)
            free_fraction = normal_cdf(
                (
                    free_end[begin:end][local_ray] - closest_depth
                )
                / longitudinal_sigma
            ).clamp(0, 1)
            hit_fraction = (
                normal_cdf(
                    (
                        hit_end[begin:end][local_ray]
                        - closest_depth
                    )
                    / longitudinal_sigma
                )
                - normal_cdf(
                    (
                        hit_start[begin:end][local_ray]
                        - closest_depth
                    )
                    / longitudinal_sigma
                )
            ).clamp(0, 1)
            behind_fraction = torch.nan_to_num(
                1.0
                - normal_cdf(
                    (
                        hit_end[begin:end][local_ray]
                        - closest_depth
                    )
                    / longitudinal_sigma
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0, 1)
            canonical_weight = (~candidate_dynamic).to(local_tau)
            canonical_hit_weight = (
                (~candidate_dynamic)
                & canonical_hit_candidate_mask[candidate_rows]
            ).to(local_tau)
            verified_canonical_weight = verified_canonical[
                candidate_rows
            ].to(local_tau)
            dynamic_weight = candidate_dynamic.to(local_tau)
            static_weight = candidate_static.to(local_tau)

            def accumulate(value, branch_weight):
                return torch.zeros(
                    ray_count, device=device, dtype=dtype
                ).scatter_add(
                    0, local_ray, value * branch_weight
                )

            canonical_free_tau_chunks.append(
                accumulate(local_tau * free_fraction, canonical_weight)
            )
            canonical_hit_tau_chunks.append(
                accumulate(
                    local_tau_hit * hit_fraction, canonical_hit_weight
                )
            )
            canonical_hit_optical_tau_chunks.append(
                accumulate(
                    local_tau_optical * hit_fraction.detach(),
                    canonical_hit_weight,
                )
            )
            canonical_behind_tau_chunks.append(
                accumulate(
                    local_tau * behind_fraction, canonical_hit_weight
                )
            )
            verified_canonical_free_tau_chunks.append(
                accumulate(
                    local_tau * free_fraction,
                    verified_canonical_weight,
                )
            )
            verified_canonical_hit_tau_chunks.append(
                accumulate(
                    local_tau_hit * hit_fraction,
                    verified_canonical_weight,
                )
            )
            verified_canonical_hit_optical_tau_chunks.append(
                accumulate(
                    local_tau_optical * hit_fraction.detach(),
                    verified_canonical_weight,
                )
            )
            verified_canonical_behind_tau_chunks.append(
                accumulate(
                    local_tau * behind_fraction,
                    verified_canonical_weight,
                )
            )
            dynamic_free_tau_chunks.append(
                accumulate(local_tau * free_fraction, dynamic_weight)
            )
            dynamic_hit_tau_chunks.append(
                accumulate(local_tau_hit * hit_fraction, dynamic_weight)
            )
            dynamic_hit_optical_tau_chunks.append(
                accumulate(
                    local_tau_optical * hit_fraction.detach(),
                    dynamic_weight,
                )
            )
            dynamic_behind_tau_chunks.append(
                accumulate(local_tau * behind_fraction, dynamic_weight)
            )
            static_hit_tau_chunks.append(
                accumulate(local_tau_hit * hit_fraction, static_weight)
            )
            static_hit_optical_tau_chunks.append(
                accumulate(
                    local_tau_optical * hit_fraction.detach(),
                    static_weight,
                )
            )
            static_behind_tau_chunks.append(
                accumulate(local_tau * behind_fraction, static_weight)
            )
        canonical_free_tau = torch.cat(canonical_free_tau_chunks)
        canonical_hit_tau = torch.cat(canonical_hit_tau_chunks)
        canonical_hit_optical_tau = torch.cat(
            canonical_hit_optical_tau_chunks
        )
        canonical_behind_tau = torch.cat(canonical_behind_tau_chunks)
        verified_canonical_free_tau = torch.cat(
            verified_canonical_free_tau_chunks
        )
        verified_canonical_hit_tau = torch.cat(
            verified_canonical_hit_tau_chunks
        )
        verified_canonical_hit_optical_tau = torch.cat(
            verified_canonical_hit_optical_tau_chunks
        )
        verified_canonical_behind_tau = torch.cat(
            verified_canonical_behind_tau_chunks
        )
        dynamic_free_tau = torch.cat(dynamic_free_tau_chunks)
        dynamic_hit_tau = torch.cat(dynamic_hit_tau_chunks)
        dynamic_hit_optical_tau = torch.cat(
            dynamic_hit_optical_tau_chunks
        )
        dynamic_behind_tau = torch.cat(dynamic_behind_tau_chunks)
        static_hit_tau = torch.cat(static_hit_tau_chunks)
        static_hit_optical_tau = torch.cat(
            static_hit_optical_tau_chunks
        )
        static_behind_tau = torch.cat(static_behind_tau_chunks)
        kind = observation_kind
        weight = self.confidence[selected].to(
            device=device, dtype=dtype
        ).clamp(0.05, 1.0)
        seed_bound = self.primitive_ids[selected].to(device=device) >= 0
        ownerless_hit = (~seed_bound) & (kind > 0)
        # Runtime consumption by an existing verified canonical lineage is
        # not canonical promotion. The producer established that lineage
        # from independent cameras; this row supplies an additional
        # same-ray interval factor without changing role/support metadata.
        ownerless_canonical_hit = ownerless_hit & (
            verified_canonical_hit_tau.detach() > 0
        )
        ownerless_dynamic_hit = ownerless_hit & (
            dynamic_hit_tau.detach() > 0
        )
        ownerless_static_hit = ownerless_hit & (
            static_hit_tau.detach() > 0
        )
        verified_ownerless = ownerless_canonical_hit | ownerless_static_hit
        ownerless_missing_dynamic_hit = (
            ownerless_hit & ~ownerless_dynamic_hit
        )
        ownerless_supported_hit = (
            ownerless_canonical_hit
            | ownerless_dynamic_hit
            | ownerless_static_hit
        )
        ownerless_missing_supported_hit = (
            ownerless_hit & ~ownerless_supported_hit
        )
        if bool(ownerless_missing_supported_hit.any()):
            proposal_rows = torch.nonzero(
                ownerless_missing_supported_hit, as_tuple=False
            ).flatten()
            midpoint = 0.5 * (
                hit_start[proposal_rows] + hit_end[proposal_rows]
            )
            proposal_centers = (
                origin[None]
                + direction_world[proposal_rows] * midpoint[:, None]
            )
            proposal_grid = torch.stack(
                [
                    2.0
                    * (render_x[proposal_rows] + 0.5)
                    / float(view.image_width)
                    - 1.0,
                    2.0
                    * (render_y[proposal_rows] + 0.5)
                    / float(view.image_height)
                    - 1.0,
                ],
                dim=-1,
            ).reshape(1, -1, 1, 2)
            proposal_image = getattr(view, "original_image", None)
            proposal_colors = None
            if proposal_image is not None:
                proposal_image = proposal_image.to(
                    device=device, dtype=dtype
                )
                proposal_colors = F.grid_sample(
                    proposal_image[None],
                    proposal_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).reshape(3, -1).T
            self._latest_uncovered_hit_proposals = {
                "centers": proposal_centers.detach().cpu(),
                "confidence": weight[proposal_rows].detach().cpu(),
                "camera_id": int(view.colmap_id),
                # Retain the complete calibrated hit interval. Static birth
                # consensus must intersect cross-sequence posterior segments;
                # requiring their arbitrary midpoints to coincide rejects
                # persistent crown volume whenever leaves move or the depth
                # posterior is broad.
                "origins": origin[None]
                .expand(len(proposal_rows), -1)
                .detach()
                .cpu(),
                "directions": direction_world[proposal_rows].detach().cpu(),
                "hit_start": hit_start[proposal_rows].detach().cpu(),
                "hit_end": hit_end[proposal_rows].detach().cpu(),
            }
            if proposal_colors is not None:
                self._latest_uncovered_hit_proposals["colors"] = (
                    proposal_colors.detach().cpu()
                )
        ownerless_free_tau = (
            (
                canonical_free_tau
                if split_candidate_permissions
                else verified_canonical_free_tau
            )
            + dynamic_free_tau
        )
        ownerless_hit_tau = (
            verified_canonical_hit_tau
            + dynamic_hit_tau
            + static_hit_tau
        )
        ownerless_behind_tau = (
            verified_canonical_behind_tau
            + dynamic_behind_tau
            + static_behind_tau
        )
        selected_free_tau = torch.where(
            seed_bound, canonical_free_tau, ownerless_free_tau
        )
        selected_hit_tau = torch.where(
            seed_bound, canonical_hit_tau, ownerless_hit_tau
        )
        # Seed-bound multi-view rows retain their canonical optical owner.
        # An ownerless observation may also route optical existence to a
        # canonical lineage that was *already* verified independently.  The
        # previous implementation allowed that lineage to explain the
        # confidence-weighted geometry term, but omitted it from the
        # complementary optical term.  In a static reconstruction (where the
        # dynamic branch is intentionally empty), most ownerless tree hits
        # therefore contributed a large constant -log(1e-4) with no opacity
        # gradient.  This path cannot promote an unrelated lineage: the
        # verified mask is established before consuming this ray.
        selected_hit_optical_tau = torch.where(
            seed_bound,
            canonical_hit_optical_tau,
            verified_canonical_hit_optical_tau
            + dynamic_hit_optical_tau
            + static_hit_optical_tau,
        )
        selected_behind_tau = torch.where(
            seed_bound, canonical_behind_tau, ownerless_behind_tau
        )
        hit_alpha = -torch.expm1(-selected_hit_tau)
        hit_alpha_optical = -torch.expm1(-selected_hit_optical_tau)
        # A hit interval also confirms that the space before it should be
        # transparent.  Penalizing pre-hit mass is what distinguishes a
        # layered crown from a broad foreground fog slab.
        free_evidence_mask = kind != 0
        confirmed_free_mask = kind < 0
        hit_mask = kind > 0
        # A confirmed-free observation has no positive owner. It is valid
        # negative evidence for both persistent canonical mass and the exact
        # sequence-local births visible from this camera.
        free_tau = torch.where(
            confirmed_free_mask,
            canonical_free_tau + dynamic_free_tau,
            selected_free_tau,
        )
        free_denominator = free_evidence_mask.sum().clamp_min(1).to(dtype)
        confirmed_free_loss = (
            (free_tau * weight)[confirmed_free_mask].sum()
            / free_denominator
            if bool(confirmed_free_mask.any())
            else zero
        )
        hit_prehit_loss = (
            (free_tau * weight)[hit_mask].sum() / free_denominator
            if bool(hit_mask.any())
            else zero
        )
        free_loss = confirmed_free_loss + hit_prehit_loss
        hit_geometry_loss = (
            (
                # Add a small observation-noise floor rather than clamping
                # the predicted mass.  clamp_min made every hit with
                # alpha<1e-5 exactly gradient-free, so an initially empty
                # canopy could never recover even when a nearby candidate
                # existed.  The additive floor keeps the likelihood finite
                # while preserving its derivative all the way to zero mass.
                -torch.log(hit_alpha + 1e-4) * weight
            )[hit_mask].sum()
            / hit_mask.sum().clamp_min(1).to(dtype)
            if bool(hit_mask.any())
            else zero
        )
        # Complete the missing confidence mass with an opacity-only
        # likelihood.  Thus depth confidence scales placement/free-space
        # gradients absolutely, while the observed tree pixel keeps one unit
        # of optical-existence evidence.  At confidence one this term is
        # exactly zero.
        optical_residual_weight = (1.0 - weight).clamp(0.0, 1.0)
        hit_optical_loss = (
            (
                -torch.log(hit_alpha_optical + 1e-4)
                * optical_residual_weight
            )[hit_mask].sum()
            / hit_mask.sum().clamp_min(1).to(dtype)
            if bool(hit_mask.any())
            else zero
        )
        hit_loss = hit_geometry_loss + hit_optical_loss
        behind_mass = (
            (
                -torch.expm1(-selected_behind_tau) * weight
            )[hit_mask].sum()
            / weight[hit_mask].sum().clamp_min(1)
            if bool(hit_mask.any())
            else zero
        )
        unknown_mask = kind == 0
        self.factor_calls += 1
        self.factor_calls_with_rays += 1
        if candidate_evaluations:
            self.interval_factor_calls_with_candidates += 1
        self.interval_candidate_evaluations += candidate_evaluations
        self.interval_canonical_candidate_evaluations += (
            canonical_candidate_evaluations
        )
        self.interval_exact_dynamic_candidate_evaluations += (
            exact_dynamic_candidate_evaluations
        )
        self.sampled_rays += int(len(selected))
        self.sampled_confirmed_free += int(
            confirmed_free_mask.sum()
        )
        self.sampled_hits += int(hit_mask.sum())
        self.sampled_unknown += int(unknown_mask.sum())
        self.sampled_ownerless_hit_rows += int(ownerless_hit.sum())
        self.ownerless_canonical_hit_rows += int(
            ownerless_canonical_hit.sum()
        )
        self.ownerless_dynamic_hit_rows += int(
            ownerless_dynamic_hit.sum()
        )
        self.ownerless_missing_dynamic_hit_rows += int(
            ownerless_missing_dynamic_hit.sum()
        )
        self.ownerless_supported_hit_rows += int(
            ownerless_supported_hit.sum()
        )
        self.ownerless_missing_supported_hit_rows += int(
            ownerless_missing_supported_hit.sum()
        )
        self.verified_ownerless_hit_rows += int(
            verified_ownerless.sum()
        )
        self.excluded_ownerless_hit_rows += int(
            ownerless_missing_supported_hit.sum()
        )
        if bool(verified_ownerless.any()):
            verified_rows = selected[verified_ownerless.detach().cpu()]
            self.ownerless_canonical_verified[verified_rows] = True
        self.consumed_camera_ids.add(int(view.colmap_id))
        audit = {
            "rays": int(len(selected)),
            "candidate_evaluations": candidate_evaluations,
            "canonical_candidate_evaluations": (
                canonical_candidate_evaluations
            ),
            "canonical_hit_candidate_evaluations": (
                canonical_hit_candidate_evaluations
            ),
            "exact_dynamic_candidate_evaluations": (
                exact_dynamic_candidate_evaluations
            ),
            "exact_static_candidate_evaluations": (
                exact_static_candidate_evaluations
            ),
            "rays_with_candidates": rays_with_candidates,
            "lookup_ray_chunk": int(lookup_chunk),
            "lookup_candidate_pool": int(len(candidate_pool)),
            "lookup_matrix_element_budget": int(
                maximum_lookup_matrix_elements
            ),
            "confirmed_free_rays": int(
                confirmed_free_mask.sum()
            ),
            "hit_rays": int(hit_mask.sum()),
            "unknown_rays": int(unknown_mask.sum()),
            "ownerless_hit_rays": int(ownerless_hit.sum()),
            "ownerless_canonical_hit_rays": int(
                ownerless_canonical_hit.sum()
            ),
            "ownerless_dynamic_hit_rays": int(
                ownerless_dynamic_hit.sum()
            ),
            "ownerless_static_hit_rays": int(
                ownerless_static_hit.sum()
            ),
            "ownerless_missing_dynamic_hit_rays": int(
                ownerless_missing_dynamic_hit.sum()
            ),
            "ownerless_supported_hit_rays": int(
                ownerless_supported_hit.sum()
            ),
            "ownerless_missing_supported_hit_rays": int(
                ownerless_missing_supported_hit.sum()
            ),
            "verified_ownerless_hit_rays": int(
                verified_ownerless.sum()
            ),
            "excluded_ownerless_hit_rays": int(
                ownerless_missing_supported_hit.sum()
            ),
            "free": float(free_loss.detach()),
            "hit": float(hit_loss.detach()),
            "hit_geometry": float(hit_geometry_loss.detach()),
            "hit_optical_existence": float(hit_optical_loss.detach()),
            "hit_opacity_gradient_scale": hit_opacity_gradient_scale,
            "mean_confidence": float(weight.mean()),
            "hit_mean_confidence": (
                float(weight[hit_mask].mean())
                if bool(hit_mask.any())
                else 0.0
            ),
            "free_mean_confidence": (
                float(weight[free_evidence_mask].mean())
                if bool(free_evidence_mask.any())
                else 0.0
            ),
            "confidence_normalization": (
                "depth_geometry_weighted_sum_over_ray_count__missing_"
                "confidence_mass_routes_to_opacity_only_existence"
            ),
            # Multi-layer foliage behind the first measured interval is
            # physically allowed, so this is an audit rather than a penalty.
            "behind_mass": float(behind_mass.detach()),
            "ownerless_hit_contract": (
                "existing_cross_sequence_canonical_lineage_plus_exact_"
                "camera_dynamic_or_fused_static_detail__never_canonical_"
                "promotion"
            ),
            "confirmed_free_contract": (
                "global_canonical_free_plus_exact_camera_dynamic__positive_"
                "canonical_hit_permission_is_independent"
                if split_candidate_permissions
                else "canonical_plus_exact_camera_dynamic"
            ),
            "depth_coordinate": "camera_z_converted_to_unit_ray_distance",
        }
        total_loss = free_loss + hit_loss
        if return_loss_components:
            return total_loss, audit, {
                "free": free_loss,
                "confirmed_free": confirmed_free_loss,
                "hit_prehit": hit_prehit_loss,
                "hit": hit_loss,
            }
        return total_loss, audit

    def factor(self, view, package, *, maximum_rays: int = 8192):
        self.factor_calls += 1
        depth_map = package.volume_depth[0]
        alpha_map = package.volume_alpha[0]
        zero = depth_map.new_zeros(())
        # Legacy rendered-depth diagnostic: retain unknown rows so this path
        # can explicitly audit that they contribute to neither free nor hit
        # loss.  The optimization path above uses only the effective free/hit
        # per-camera epoch and owns ``interval_row_visits``.
        selected = torch.nonzero(
            self.camera_ids == int(view.colmap_id), as_tuple=False
        ).flatten()
        if not len(selected):
            return zero, {
                "rays": 0,
                "confirmed_free_rays": 0,
                "hit_rays": 0,
                "unknown_rays": 0,
                "in_bounds_rays": 0,
                "valid_depth_rays": 0,
                "free": 0.0,
                "hit": 0.0,
            }
        if len(selected) > int(maximum_rays):
            start = (
                (self.factor_calls - 1) * int(maximum_rays)
            ) % len(selected)
            positions = (
                torch.arange(int(maximum_rays)) + start
            ) % len(selected)
            selected = selected[positions]
        device, dtype = depth_map.device, depth_map.dtype
        pixels = self.pixels[selected].to(device=device, dtype=dtype)
        source_size = self.source_image_sizes[selected].to(
            device=device, dtype=dtype
        )
        render_x = pixels[:, 0] * (
            float(view.image_width) / source_size[:, 0]
        )
        render_y = pixels[:, 1] * (
            float(view.image_height) / source_size[:, 1]
        )
        grid = torch.stack(
            [
                2.0 * (render_x + 0.5) / view.image_width - 1.0,
                2.0 * (render_y + 0.5) / view.image_height - 1.0,
            ],
            dim=-1,
        ).reshape(1, -1, 1, 2)
        depth = F.grid_sample(
            depth_map[None, None], grid, align_corners=False
        ).reshape(-1)
        alpha = F.grid_sample(
            alpha_map[None, None], grid, align_corners=False
        ).reshape(-1)
        free_end = self.free_end[selected].to(device=device, dtype=dtype)
        hit_start = self.hit_start[selected].to(device=device, dtype=dtype)
        hit_end = self.hit_end[selected].to(device=device, dtype=dtype)
        kind = self.observation_type[selected].to(device=device)
        weight = self.confidence[selected].to(
            device=device, dtype=dtype
        ).clamp(0.05, 1.0)
        has_mass = (depth > 0).to(dtype)
        free_penalty = (
            (free_end - depth).clamp_min(0)
            / free_end.clamp_min(0.1)
            * alpha
            * has_mass
        )
        # -1 is confirmed free space, +1 is a foliage hit, and 0 is
        # deliberately unknown (occluded, behind the measured interval, or
        # otherwise ambiguous).  Treating ``0`` as free carved valid canopy
        # from every ambiguous ray and was a direct cause of the sparse tree
        # branch.
        free_mask = kind < 0
        free_loss = (
            (free_penalty * weight)[free_mask].sum()
            / weight[free_mask].sum().clamp_min(1)
            if bool(free_mask.any())
            else zero
        )
        hit_mask = kind > 0
        if bool(hit_mask.any()):
            interval_distance = (
                (hit_start - depth).clamp_min(0)
                + (depth - hit_end).clamp_min(0)
            ) / (hit_end - hit_start).clamp_min(0.1)
            hit_penalty = -torch.log(alpha.clamp_min(1e-4)) + interval_distance
            hit_loss = (
                (hit_penalty * weight)[hit_mask].sum()
                / weight[hit_mask].sum().clamp_min(1)
            )
        else:
            hit_loss = zero
        unknown_mask = kind == 0
        in_bounds = (
            torch.isfinite(grid).all(dim=-1).reshape(-1)
            & (grid.reshape(-1, 2).abs() <= 1.0).all(dim=-1)
        )
        self.factor_calls_with_rays += 1
        self.sampled_rays += int(len(selected))
        self.sampled_confirmed_free += int(free_mask.sum())
        self.sampled_hits += int(hit_mask.sum())
        self.sampled_unknown += int(unknown_mask.sum())
        self.consumed_camera_ids.add(int(view.colmap_id))
        return free_loss + hit_loss, {
            "rays": int(len(selected)),
            "confirmed_free_rays": int(free_mask.sum()),
            "hit_rays": int(hit_mask.sum()),
            "unknown_rays": int(unknown_mask.sum()),
            "in_bounds_rays": int(in_bounds.sum()),
            "valid_depth_rays": int((depth > 0).sum()),
            "free": float(free_loss.detach()),
            "hit": float(hit_loss.detach()),
        }


def _rigid_global_track_mask(archive) -> np.ndarray:
    """Select only cross-traversal rigid tracks for surface ray factors."""
    probability = np.asarray(archive["role_probabilities"])
    keep = probability[:, 0] >= 0.75
    if probability.shape[1] > 1:
        keep &= probability[:, 1] <= 0.15
    if "sequence_count" in archive:
        keep &= np.asarray(archive["sequence_count"]) >= 2
    if "cycle_consistency" in archive:
        keep &= np.asarray(archive["cycle_consistency"]) >= 0.50
    if "pointmap_spread" in archive:
        keep &= np.asarray(archive["pointmap_spread"]) <= 0.08
    return keep


class OutdoorGeometryEvidence:
    """Keep every geometry source separate at loss construction time."""

    def __init__(
        self,
        evidence_store: Path,
        *,
        chart_base_source: str = "matcha",
    ):
        if chart_base_source not in MOGE3_CHART_BASE_SOURCES:
            raise ValueError(
                f"Unsupported Chart base source: {chart_base_source!r}"
            )
        self.chart_base_source = str(chart_base_source)
        self.store = load_evidence_store(evidence_store, verify_hashes=False)
        chart_path = artifact_path(
            self.store, "chart_geometry", required=False
        )
        chart_camera_path = artifact_path(
            self.store, "chart_cameras", required=False
        )
        self.chart = None
        self.moge3_chart_base_metadata: dict = {}
        self.moge3_chart_scale_audit: dict = {}
        self.chart_scale_factor = 1.0
        self.frame_by_stem: dict[str, int] = {}
        self.pointmap_records: dict[str, dict] = {}
        self._pointmap_cache: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._pointmap_posterior_cache: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._pointmap_posterior_stats = {
            "factor_calls": 0,
            "posterior_factor_calls": 0,
            "single_sequence_fallback_factor_calls": 0,
            "pixels": 0,
            "cross_sequence_supported_pixels": 0,
            "single_sequence_low_precision_pixels": 0,
            "missing_posterior_fallback_pixels": 0,
            "precision_sum": 0.0,
        }
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
                "precision": np.ones(
                    archive["confs"].shape, dtype=np.float32
                ),
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
            chart_base_path = artifact_path(
                self.store, "moge3_chart_base", required=False
            )
            if chart_base_path is not None:
                chart_contract = load_chart_base_metadata(
                    chart_base_path,
                    expected_names=cameras["filepaths"],
                )
                self.moge3_chart_base_metadata = dict(
                    chart_contract["metadata"]
                )
                self.moge3_chart_scale_audit = dict(
                    self.moge3_chart_base_metadata.get(
                        "scale_calibration", {}
                    )
                )
                chart_scale = self.moge3_chart_scale_audit.get(
                    "metric_to_cambridge_scale"
                )
                chart_sigma = self.moge3_chart_scale_audit.get(
                    "global_log_scale_sigma"
                )
                if (
                    chart_scale is None
                    or not np.isfinite(float(chart_scale))
                    or float(chart_scale) <= 0.0
                    or chart_sigma is None
                    or not np.isfinite(float(chart_sigma))
                    or float(chart_sigma) < 0.0
                ):
                    raise RuntimeError(
                        "MoGe3 Chart evidence has no valid scene-scale contract"
                    )
            consensus_path = artifact_path(
                self.store,
                "chart_crossview_consensus",
                required=False,
            )
            if (
                mast3r_is_geometry_authority(self.store)
                and consensus_path is None
            ):
                raise RuntimeError(
                    "MASt3R-only Teacher requires pixelwise Chart "
                    "cross-view consensus"
                )
            if consensus_path is not None:
                with np.load(consensus_path, allow_pickle=False) as consensus:
                    required = {
                        "depths",
                        "support_counts",
                        "consistency_weights",
                        "correction_mask",
                        "image_names",
                    }
                    missing = required - set(consensus.files)
                    if missing:
                        raise RuntimeError(
                            "Chart consensus lacks required arrays: "
                            + ", ".join(sorted(missing))
                        )
                    names = [
                        Path(str(value)).stem
                        for value in consensus["image_names"]
                    ]
                    expected_names = [
                        Path(str(value)).stem
                        for value in cameras["filepaths"]
                    ]
                    if names != expected_names:
                        raise RuntimeError(
                            "Chart consensus camera order does not match atlas"
                        )
                    support = consensus["support_counts"].astype(np.uint8)
                    accepted = consensus["correction_mask"].astype(bool)
                    consistency = consensus[
                        "consistency_weights"
                    ].astype(np.float32)
                    minimum_support = int(
                        consensus["minimum_consensus_views"].item()
                        if "minimum_consensus_views" in consensus
                        else 2
                    )
                if support.shape != self.chart["depth"].shape:
                    raise RuntimeError(
                        "Chart consensus shape does not match Chart depth"
                    )
                confirmed = accepted & (support >= minimum_support)
                contradicted = (support >= minimum_support) & ~accepted
                # Preserve MAtCha's own aligned Chart depth.  The former code
                # unconditionally replaced it with a Chart-to-Chart consensus
                # depth, although an independent audit showed that only 45.9%
                # of those corrections moved closer to reliable geometry.
                # Consensus now changes precision, not the measurement, until
                # independently corroborated correction pixels are persisted
                # by the evidence builder.
                trust = np.full(
                    support.shape, 0.15, dtype=np.float32
                )
                trust[confirmed] = np.clip(
                    consistency[confirmed], 0.10, 1.0
                )
                trust[contradicted] = 0.0
                self.chart["precision"] = trust
                self.chart["reference_mask"] &= ~contradicted
                self.chart["crossview_support"] = support
                self.chart["crossview_confirmed"] = confirmed
                self.chart["consensus_depth_override_applied"] = False
            if self.chart_base_source != "matcha":
                if chart_base_path is None:
                    raise RuntimeError(
                        f"Chart base {self.chart_base_source!r} requires the "
                        "moge3_chart_base evidence artifact"
                    )
                base = load_chart_base(
                    chart_base_path,
                    source=self.chart_base_source,
                    expected_names=cameras["filepaths"],
                )
                if base["depth"].shape != self.chart["depth"].shape:
                    raise RuntimeError(
                        "MoGe3 Chart base shape does not match MAtCha"
                    )
                self.chart["depth"] = base["depth"]
                self.chart["reference_mask"] &= base["validity"]
                self.chart["precision"] *= base["precision"]
                self.chart["moge3_base_metadata"] = base["metadata"]
        pointmap_index_path = artifact_path(
            self.store, "mast3r_pointmap_index", required=False
        )
        if pointmap_index_path is not None:
            pointmap_index = json.loads(
                pointmap_index_path.read_text(encoding="utf-8")
            )
            if (
                pointmap_index.get("coordinate_frame")
                != "cambridge_fixed_world"
            ):
                raise RuntimeError(
                    "MASt3R pointmaps are not in the fixed Cambridge world"
                )
            camera_order = pointmap_index.get("camera_order", [])
            records = pointmap_index.get("records", {})
            if set(camera_order) != set(records):
                raise RuntimeError(
                    "MASt3R pointmap camera order does not cover its records"
                )
            for stem in camera_order:
                record = dict(records[stem])
                path = Path(record["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(record["bytes"])
                ):
                    raise RuntimeError(
                        "Indexed MASt3R pointmap changed or disappeared: "
                        f"{path}"
                    )
                record["path"] = path
                posterior = record.get("cross_sequence_posterior")
                if posterior is not None:
                    posterior = dict(posterior)
                    posterior_path = Path(posterior["path"])
                    if (
                        not posterior_path.is_file()
                        or posterior_path.stat().st_size
                        != int(posterior["bytes"])
                    ):
                        raise RuntimeError(
                            "Indexed MASt3R cross-sequence posterior changed "
                            f"or disappeared: {posterior_path}"
                        )
                    posterior["path"] = posterior_path
                    record["cross_sequence_posterior"] = posterior
                self.pointmap_records[str(stem)] = record
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
        moge3_index_path = artifact_path(
            self.store, "moge3_index", required=False
        )
        moge3_runtime_cache_path = artifact_path(
            self.store, "moge3_runtime_cache", required=False
        )
        self.moge3_records: dict[str, dict] = {}
        self.moge3_runtime_cache: dict | None = None
        self.moge3_metric_to_cambridge_scale: float | None = None
        # Loading raw metric fields before the scene gauge is bound would
        # silently mix coordinate systems. Placeholder foliage used by the
        # causal rigid pretrain intentionally leaves this disabled.
        self.moge3_runtime_enabled = False
        self._verified_moge3_stems: set[str] = set()
        self._moge3_cache: OrderedDict[str, dict[str, np.ndarray]] = (
            OrderedDict()
        )
        self._moge3_cache_maximum_views = 4
        if moge3_index_path is not None:
            moge3_index = load_moge3_index(
                moge3_index_path, verify_views=False
            )
            self.moge3_records = {
                str(stem): dict(record)
                for stem, record in moge3_index["records"].items()
            }
            if moge3_runtime_cache_path is not None:
                self.moge3_runtime_cache = load_moge3_runtime_cache(
                    moge3_runtime_cache_path,
                    expected_source_index_sha256=moge3_sha256_file(
                        moge3_index_path
                    ),
                    verify_arrays=True,
                )
                if self.moge3_runtime_cache["camera_order"] != list(
                    moge3_index["camera_order"]
                ):
                    raise RuntimeError(
                        "MoGe3 runtime cache camera order changed"
                    )
        elif moge3_runtime_cache_path is not None:
            raise RuntimeError(
                "MoGe3 runtime cache cannot exist without moge3_index"
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
        self.track_observations: dict[str, dict[str, np.ndarray]] = {}
        self.track_graph_contract = {
            "representation": "source_pointmap_projection_star_tracks",
            "descriptor_reciprocal_union_find_tracks": False,
            "camera_scope": "selected_matcha_chart_views",
        }
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
                if source_code == 1:
                    schema = (
                        str(archive["schema_version"].item())
                        if "schema_version" in archive
                        else None
                    )
                    if schema != TRACK_GRAPH_VERSION:
                        raise RuntimeError(
                            "Training evidence contains a stale MASt3R track "
                            f"graph {schema!r}; expected "
                            f"{TRACK_GRAPH_VERSION!r}. Rebuild evidence and "
                            "role-aware initialization."
                        )
                    if "correspondence_builder" in archive:
                        builder = str(
                            archive["correspondence_builder"].item()
                        )
                        descriptor = (
                            builder
                            == "mast3r_reciprocal_descriptor_union_find"
                        )
                        self.track_graph_contract = {
                            "representation": builder,
                            "descriptor_reciprocal_union_find_tracks": (
                                descriptor
                            ),
                            "camera_scope": (
                                str(archive["camera_scope"].item())
                                if "camera_scope" in archive
                                else "unspecified"
                            ),
                        }
                order = np.argsort(archive["track_id"])
                self.track_archives[source_code] = {
                    "track_id": archive["track_id"][order].astype(np.int64),
                    "xyz": archive["xyz"][order].astype(np.float32),
                    "variance": archive["position_covariance_diag"][
                        order
                    ].astype(np.float32),
                }
                required_observations = {
                    "observation_camera_indices",
                    "observation_pixels",
                    "observation_camera_depth",
                    "observation_confidence",
                    "camera_names",
                    "camera_image_sizes",
                }
                if source_code == 1:
                    missing = sorted(
                        required_observations - set(archive.files)
                    )
                    if missing:
                        raise RuntimeError(
                            "MASt3R training evidence lacks observation-level "
                            "ray factors: " + ", ".join(missing)
                        )
                    camera_indices = archive[
                        "observation_camera_indices"
                    ].astype(np.int32)
                    pixels = archive["observation_pixels"].astype(np.float32)
                    depths = archive[
                        "observation_camera_depth"
                    ].astype(np.float32)
                    confidence = archive[
                        "observation_confidence"
                    ].astype(np.float32)
                    sizes = archive["camera_image_sizes"].astype(np.int32)
                    names = [Path(str(value)).stem for value in archive["camera_names"]]
                    offsets = archive["observation_offsets"].astype(np.int64)
                    track_rigid = _rigid_global_track_mask(archive)
                    observation_rigid = np.repeat(
                        track_rigid, np.diff(offsets)
                    )
                    if len(observation_rigid) != len(camera_indices):
                        raise RuntimeError(
                            "Track observation offsets do not cover flat arrays"
                        )
                    for camera_index, name in enumerate(names):
                        selected = (
                            (camera_indices == camera_index)
                            & observation_rigid
                        )
                        self.track_observations[name] = {
                            "pixels": pixels[selected],
                            "depth": depths[selected],
                            "confidence": confidence[selected],
                            "image_size": sizes[camera_index],
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
        self._track_samplers = {
            source: EvidenceEpochSampler(
                len(archive["track_id"]), seed=1701 + int(source)
            )
            for source, archive in self.track_archives.items()
        }
        self._structure_sampler = (
            None
            if self.structure is None
            else EvidenceEpochSampler(
                len(self.structure["centers"]), seed=2719
            )
        )
        self.consumed = {
            "chart": 0,
            "chart_native_factor": 0,
            "mast3r_pointmap_native_factor": 0,
            "plane": 0,
            "inverse_depth": 0,
            "moge3": 0,
            "dav2": 0,
            "track_factor": 0,
            "track_observation_factor": 0,
            "structure_factor": 0,
        }
        self.preflight = self._metric_preflight()

    def configure_moge3_metric_scale(self, value: float) -> None:
        """Bind raw MoGe metric depth to the initialized Cambridge gauge."""
        scale = float(value)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("MoGe3 metric-to-Cambridge scale must be positive")
        self.moge3_metric_to_cambridge_scale = scale
        self.moge3_runtime_enabled = True

    def chart_native_factor(
        self,
        image_name: str,
        package,
        rigid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compare rendered inverse depth at the native Chart resolution.

        MAtCha does not provide high-resolution measurements.  Upsampling its
        depth and validity to the RGB render resolution invents sharp
        supervision at facade boundaries.  Instead, aggregate rendered
        surface inverse depth and task responsibility into the native atlas
        cells and compare only there.
        """
        depth_map = package.surface_depth[0]
        alpha_map = package.surface_alpha[0]
        zero = depth_map.new_zeros(())
        stem = Path(str(image_name)).stem
        frame = self.frame_by_stem.get(stem)
        if (
            frame is None
            or self.chart is None
            or not self.chart["active"][frame]
        ):
            return zero, {"pixels": 0, "height": 0, "width": 0}
        target_np = self.chart["depth"][frame]
        confidence_np = self.chart["confidence"][frame]
        precision_archive = self.chart.get("precision")
        precision_np = (
            np.ones_like(confidence_np, dtype=np.float32)
            if precision_archive is None
            else precision_archive[frame]
        )
        valid_np = (
            self.chart["reference_mask"][frame]
            & np.isfinite(target_np)
            & (target_np > 0)
            & np.isfinite(confidence_np)
            & (confidence_np > 0)
            & np.isfinite(precision_np)
            & (precision_np > 0)
        )
        if not bool(valid_np.any()):
            return zero, {"pixels": 0, "height": 0, "width": 0}
        native_shape = tuple(map(int, target_np.shape))
        device, dtype = depth_map.device, depth_map.dtype
        target = torch.from_numpy(
            np.array(target_np, copy=True, order="C")
        ).to(device=device, dtype=dtype)
        confidence = torch.from_numpy(
            np.array(confidence_np, copy=True, order="C")
        ).to(device=device, dtype=dtype)
        precision = torch.from_numpy(
            np.array(precision_np, copy=True, order="C")
        ).to(device=device, dtype=dtype)
        valid_target = torch.from_numpy(
            np.array(valid_np, copy=True, order="C")
        ).to(device=device)
        safe_depth = torch.where(
            torch.isfinite(depth_map) & (depth_map > 0),
            depth_map,
            torch.ones_like(depth_map),
        )
        render_weight = (
            alpha_map.clamp(0, 1)
            * rigid.clamp(0, 1)
            * (torch.isfinite(depth_map) & (depth_map > 0)).to(dtype)
        )
        weighted_rho = F.interpolate(
            ((safe_depth.reciprocal() * render_weight)[None, None]),
            size=native_shape,
            mode="area",
        )[0, 0]
        native_weight = F.interpolate(
            render_weight[None, None],
            size=native_shape,
            mode="area",
        )[0, 0]
        predicted_rho = weighted_rho / native_weight.clamp_min(1e-4)
        target_rho = target.clamp_min(1e-4).reciprocal()
        # Reject native cells straddling a strong Chart discontinuity.  These
        # cells do not describe a single surface posterior.
        horizontal = F.pad(
            (target_rho[:, 1:] - target_rho[:, :-1]).abs(),
            (0, 1, 0, 0),
        )
        vertical = F.pad(
            (target_rho[1:, :] - target_rho[:-1, :]).abs(),
            (0, 0, 0, 1),
        )
        discontinuity = torch.maximum(horizontal, vertical)
        edge_valid = discontinuity <= 0.12 * target_rho.clamp_min(1e-4)
        supervision_valid = valid_target & edge_valid
        if not bool(supervision_valid.any()):
            return zero, {
                "pixels": 0,
                "depth_pixels": 0,
                "height": native_shape[0],
                "width": native_shape[1],
            }
        depth_valid = supervision_valid & (native_weight >= 0.01)
        normalized_confidence = (
            confidence / confidence[valid_target].median().clamp_min(1e-6)
        ).clamp(0.1, 4.0) * precision.clamp(0.0, 1.0)
        relative = (
            (predicted_rho - target_rho).abs()
            / target_rho.clamp_min(1e-4)
        )
        weight = normalized_confidence * native_weight
        loss = (
            (
                torch.log1p(relative[depth_valid])
                * weight[depth_valid]
            ).sum()
            / weight[depth_valid].sum().clamp_min(1)
            if bool(depth_valid.any())
            else zero
        )
        # Missing rendered coverage is itself evidence failure.  The former
        # ``native_weight >= .05`` mask dropped exactly those cells from both
        # terms, so a sparse point-cloud render could report a small Chart
        # loss simply by leaving most of the atlas empty.
        hit = (
            -torch.log(
                native_weight[supervision_valid].clamp_min(1e-4)
            )
            * normalized_confidence[supervision_valid]
        ).sum() / normalized_confidence[
            supervision_valid
        ].sum().clamp_min(1)
        # Per-pixel normalized reductions intentionally remove arbitrary
        # MASt3R/MAtCha confidence scale, but must not erase the absolute
        # consensus trust assigned to the whole Chart.
        absolute_precision = precision[
            supervision_valid
        ].mean().clamp(0.0, 1.0)
        self.consumed["chart_native_factor"] += 1
        return absolute_precision * (loss + 0.03 * hit), {
            "pixels": int(supervision_valid.sum()),
            "depth_pixels": int(depth_valid.sum()),
            "height": native_shape[0],
            "width": native_shape[1],
            "depth": float(loss.detach()),
            "hit": float(hit.detach()),
            "absolute_precision": float(
                absolute_precision.detach()
            ),
        }

    def mast3r_pointmap_native_factor(
        self,
        view,
        package,
        rigid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Supervise the rendered surface on dense real MASt3R rays.

        Pointmaps are stored in the immutable Cambridge world.  They are
        projected back to their exact fixed camera here, at native evidence
        resolution, so topology may split/retire freely without severing the
        measurement from the rendered surface.
        """
        depth_map = package.surface_depth[0]
        alpha_map = package.surface_alpha[0]
        zero = depth_map.new_zeros(())
        stem = Path(str(view.image_name)).stem
        record = self.pointmap_records.get(stem)
        if record is None:
            return zero, {"pixels": 0, "height": 0, "width": 0}
        cached = self._pointmap_cache.get(stem)
        if cached is None:
            path = Path(record["path"])
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise RuntimeError(
                    f"MASt3R pointmap content hash mismatch: {path}"
                )
            cached = _load_mast3r_pointmap_geometry(path)
            self._pointmap_cache[stem] = cached
        points_flat, confidence_flat = cached
        native_shape = _native_pointmap_shape(
            len(confidence_flat),
            (int(view.image_width), int(view.image_height)),
        )
        rotation = np.asarray(view.R, dtype=np.float32)
        translation = np.asarray(view.T, dtype=np.float32)
        target_np = (
            points_flat @ rotation + translation[None]
        )[:, 2].reshape(native_shape)
        confidence_np = confidence_flat.reshape(native_shape)
        posterior_record = record.get("cross_sequence_posterior")
        precision_np = np.full(
            native_shape,
            SINGLE_SEQUENCE_POINTMAP_PRECISION,
            dtype=np.float32,
        )
        cross_sequence_supported_np = np.zeros(native_shape, dtype=bool)
        if posterior_record is not None:
            cached_posterior = getattr(
                self, "_pointmap_posterior_cache", {}
            ).get(stem)
            if cached_posterior is None:
                posterior_path = Path(posterior_record["path"])
                digest = hashlib.sha256()
                with posterior_path.open("rb") as handle:
                    for block in iter(
                        lambda: handle.read(1 << 20), b""
                    ):
                        digest.update(block)
                if digest.hexdigest() != posterior_record["sha256"]:
                    raise RuntimeError(
                        "MASt3R pointmap posterior content hash mismatch: "
                        f"{posterior_path}"
                    )
                with np.load(
                    posterior_path, allow_pickle=False
                ) as posterior:
                    schema = str(posterior["schema_version"].item())
                    if schema != (
                        "mast3r-cross-sequence-pointmap-posterior-v1"
                    ):
                        raise RuntimeError(
                            "Unsupported MASt3R pointmap posterior: "
                            f"{schema!r}"
                        )
                    precision_np = posterior["precision"].astype(np.float32)
                    cross_sequence_supported_np = posterior[
                        "cross_sequence_supported"
                    ].astype(bool)
                if (
                    tuple(precision_np.shape) != tuple(native_shape)
                    or tuple(cross_sequence_supported_np.shape)
                    != tuple(native_shape)
                ):
                    raise RuntimeError(
                        "MASt3R pointmap posterior arrays do not match "
                        "native raster: "
                        f"precision={precision_np.shape}, support="
                        f"{cross_sequence_supported_np.shape}, expected="
                        f"{native_shape}"
                    )
                precision_np = np.nan_to_num(
                    precision_np,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clip(0.0, 1.0)
                cross_sequence_supported_np &= precision_np > 0
                self._pointmap_posterior_cache[stem] = (
                    precision_np,
                    cross_sequence_supported_np,
                )
            else:
                precision_np, cross_sequence_supported_np = cached_posterior
        valid_np = (
            np.isfinite(target_np)
            & (target_np > 0)
            & np.isfinite(confidence_np)
            & (confidence_np >= 1.25)
            & np.isfinite(precision_np)
            & (precision_np > 0)
        )
        if bool(valid_np.any()):
            median_depth = float(np.median(target_np[valid_np]))
            valid_np &= target_np <= max(6.0 * median_depth, 5.0)
        if not bool(valid_np.any()):
            return zero, {
                "pixels": 0,
                "height": native_shape[0],
                "width": native_shape[1],
            }
        device, dtype = depth_map.device, depth_map.dtype
        target = torch.from_numpy(
            np.array(target_np, copy=True, order="C")
        ).to(device=device, dtype=dtype)
        confidence = torch.from_numpy(
            np.array(confidence_np, copy=True, order="C")
        ).to(device=device, dtype=dtype)
        precision = torch.from_numpy(
            np.array(precision_np, copy=True, order="C")
        ).to(device=device, dtype=dtype)
        valid_target = torch.from_numpy(
            np.array(valid_np, copy=True, order="C")
        ).to(device=device)
        cross_sequence_supported = torch.from_numpy(
            np.array(
                cross_sequence_supported_np, copy=True, order="C"
            )
        ).to(device=device)
        native_rigid = F.interpolate(
            rigid.clamp(0, 1)[None, None],
            size=native_shape,
            mode="area",
        )[0, 0]
        valid_target &= native_rigid >= 0.80
        if not bool(valid_target.any()):
            return zero, {
                "pixels": 0,
                "height": native_shape[0],
                "width": native_shape[1],
            }
        safe_depth = torch.where(
            torch.isfinite(depth_map) & (depth_map > 0),
            depth_map,
            torch.ones_like(depth_map),
        )
        render_weight = (
            alpha_map.clamp(0, 1)
            * rigid.clamp(0, 1)
            * (torch.isfinite(depth_map) & (depth_map > 0)).to(dtype)
        )
        weighted_rho = F.interpolate(
            (
                safe_depth.reciprocal()
                * render_weight
            )[None, None],
            size=native_shape,
            mode="area",
        )[0, 0]
        native_weight = F.interpolate(
            render_weight[None, None],
            size=native_shape,
            mode="area",
        )[0, 0]
        predicted_rho = weighted_rho / native_weight.clamp_min(1e-4)
        target_rho = target.clamp_min(1e-4).reciprocal()
        horizontal = F.pad(
            (target_rho[:, 1:] - target_rho[:, :-1]).abs(),
            (0, 1, 0, 0),
        )
        vertical = F.pad(
            (target_rho[1:, :] - target_rho[:-1, :]).abs(),
            (0, 0, 0, 1),
        )
        edge_valid = torch.maximum(horizontal, vertical) <= (
            0.12 * target_rho.clamp_min(1e-4)
        )
        supervision_valid = valid_target & edge_valid
        if not bool(supervision_valid.any()):
            return zero, {
                "pixels": 0,
                "height": native_shape[0],
                "width": native_shape[1],
            }
        confidence_median = confidence[valid_target].median().clamp_min(
            1e-6
        )
        normalized_confidence = (
            confidence / confidence_median
        ).clamp(0.10, 4.0) * precision.clamp(0.0, 1.0)
        depth_valid = supervision_valid & (native_weight >= 0.01)
        relative = (
            (predicted_rho - target_rho).abs()
            / target_rho.clamp_min(1e-4)
        )
        depth_weight = normalized_confidence * native_weight
        depth_loss = (
            (
                torch.log1p(relative[depth_valid])
                * depth_weight[depth_valid]
            ).sum()
            / depth_weight[depth_valid].sum().clamp_min(1)
            if bool(depth_valid.any())
            else zero
        )
        hit_weight = normalized_confidence[supervision_valid]
        hit_loss = (
            -torch.log(
                native_weight[supervision_valid].clamp_min(1e-4)
            )
            * hit_weight
        ).sum() / hit_weight.sum().clamp_min(1)
        absolute_precision = precision[
            supervision_valid
        ].mean().clamp(0.0, 1.0)
        stats = getattr(self, "_pointmap_posterior_stats", None)
        if stats is not None:
            for name, initial in (
                ("factor_calls", 0),
                ("posterior_factor_calls", 0),
                ("single_sequence_fallback_factor_calls", 0),
                ("pixels", 0),
                ("cross_sequence_supported_pixels", 0),
                ("single_sequence_low_precision_pixels", 0),
                ("missing_posterior_fallback_pixels", 0),
                ("precision_sum", 0.0),
            ):
                stats.setdefault(name, initial)
            stats["factor_calls"] += 1
            if posterior_record is None:
                stats["single_sequence_fallback_factor_calls"] += 1
            else:
                stats["posterior_factor_calls"] += 1
            stats["pixels"] += int(supervision_valid.sum())
            stats["cross_sequence_supported_pixels"] += int(
                (supervision_valid & cross_sequence_supported).sum()
            )
            unsupported_pixels = int(
                (supervision_valid & ~cross_sequence_supported).sum()
            )
            stats["single_sequence_low_precision_pixels"] += (
                unsupported_pixels
            )
            if posterior_record is None:
                stats["missing_posterior_fallback_pixels"] += (
                    unsupported_pixels
                )
            stats["precision_sum"] += float(
                precision[supervision_valid].sum().detach()
            )
        self.consumed["mast3r_pointmap_native_factor"] += 1
        return absolute_precision * (depth_loss + 0.03 * hit_loss), {
            "pixels": int(supervision_valid.sum()),
            "depth_pixels": int(depth_valid.sum()),
            "height": native_shape[0],
            "width": native_shape[1],
            "depth": float(depth_loss.detach()),
            "hit": float(hit_loss.detach()),
            "absolute_precision": float(absolute_precision.detach()),
            "cross_sequence_supported_pixels": int(
                (supervision_valid & cross_sequence_supported).sum()
            ),
            "single_sequence_low_precision_pixels": int(
                (supervision_valid & ~cross_sequence_supported).sum()
            ),
            "missing_posterior_fallback_pixels": (
                int((supervision_valid & ~cross_sequence_supported).sum())
                if posterior_record is None
                else 0
            ),
            "posterior_available": posterior_record is not None,
        }

    def track_observation_factor(
        self,
        image_name: str,
        package,
        *,
        maximum_observations: int = 8192,
    ) -> tuple[torch.Tensor, dict]:
        """Constrain the rendered surface on real fixed-camera track rays."""
        depth_map = package.surface_depth[0]
        alpha_map = package.surface_alpha[0]
        zero = depth_map.new_zeros(())
        record = self.track_observations.get(Path(str(image_name)).stem)
        if record is None or not len(record["pixels"]):
            return zero, {"matched": 0}
        count = min(int(maximum_observations), len(record["pixels"]))
        # Deterministic bounded coverage within the current camera.
        if count < len(record["pixels"]):
            indices = np.linspace(
                0, len(record["pixels"]) - 1, count, dtype=np.int64
            )
        else:
            indices = np.arange(count, dtype=np.int64)
        pixels = torch.from_numpy(record["pixels"][indices]).to(
            device=depth_map.device, dtype=depth_map.dtype
        )
        target = torch.from_numpy(record["depth"][indices]).to(
            device=depth_map.device, dtype=depth_map.dtype
        )
        confidence = torch.from_numpy(record["confidence"][indices]).to(
            device=depth_map.device, dtype=depth_map.dtype
        )
        source_width, source_height = map(int, record["image_size"])
        x = pixels[:, 0] * (depth_map.shape[1] / source_width)
        y = pixels[:, 1] * (depth_map.shape[0] / source_height)
        grid = torch.stack(
            [
                2.0 * (x + 0.5) / depth_map.shape[1] - 1.0,
                2.0 * (y + 0.5) / depth_map.shape[0] - 1.0,
            ],
            dim=-1,
        ).reshape(1, -1, 1, 2)
        predicted = F.grid_sample(
            depth_map[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(-1)
        alpha = F.grid_sample(
            alpha_map[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(-1)
        valid_target = torch.isfinite(target) & (target > 0)
        if not bool(valid_target.any()):
            return zero, {"matched": 0}
        valid_depth = (
            valid_target
            & torch.isfinite(predicted)
            & (predicted > 0)
        )
        weight = (
            confidence[valid_target]
            / confidence[valid_target].median().clamp_min(1e-6)
        ).clamp(0.25, 4.0)
        if bool(valid_depth.any()):
            depth_weight = (
                confidence[valid_depth]
                / confidence[valid_target].median().clamp_min(1e-6)
            ).clamp(0.25, 4.0)
            relative = (
                (predicted[valid_depth] - target[valid_depth]).abs()
                / target[valid_depth].clamp_min(0.1)
            )
            depth_loss = (
                torch.log1p(relative) * depth_weight
            ).sum() / depth_weight.sum().clamp_min(1)
        else:
            depth_loss = zero
        # Apply the observation hit term to every valid track ray, including
        # holes with zero rendered depth.  Filtering it with ``valid_depth``
        # made the observation factor reward already-covered pixels only.
        hit_loss = (
            -torch.log(alpha[valid_target].clamp_min(1e-4)) * weight
        ).sum() / weight.sum().clamp_min(1)
        self.consumed["track_observation_factor"] += 1
        return depth_loss + 0.05 * hit_loss, {
            "matched": int(valid_target.sum()),
            "depth_matched": int(valid_depth.sum()),
            "depth": float(depth_loss.detach()),
            "hit": float(hit_loss.detach()),
        }

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
        """Persistent external track factor with shuffled full coverage.

        A track remains an evidence node after its original render primitive
        is cloned, split, or retired. Descendants may share the same track id;
        their losses are normalized per external track so topology cannot
        multiply a measurement's weight.
        """
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
        sampled_total = 0
        for source, archive in self._track_device_cache[device_key].items():
            sampled_numpy = self._track_samplers[source].next(
                maximum_tracks
            )
            sampled_total += int(len(sampled_numpy))
            sampled = torch.from_numpy(sampled_numpy).to(
                device=device, dtype=torch.long
            )
            selected_archive = torch.zeros(
                len(archive["track_id"]), dtype=torch.bool, device=device
            )
            selected_archive[sampled] = True
            candidate = (
                (surface._source_type == int(source))
                & (surface._track_id >= 0)
            )
            indices = torch.nonzero(candidate, as_tuple=False).flatten()
            if not len(indices):
                continue
            ids = surface._track_id[indices]
            position = torch.searchsorted(archive["track_id"], ids)
            in_range = position < len(archive["track_id"])
            safe_position = position.clamp_max(
                max(len(archive["track_id"]) - 1, 0)
            )
            matched = in_range & (
                archive["track_id"][safe_position] == ids
            ) & selected_archive[safe_position]
            if not bool(matched.any()):
                continue
            indices = indices[matched]
            position = safe_position[matched]
            variance = archive["variance"][position].clamp_min(1e-6)
            delta = surface.get_xyz[indices] - archive["xyz"][position]
            mahalanobis = (delta.square() / variance).sum(-1)
            per_primitive = torch.log1p(mahalanobis)
            # One local primitive explaining the external track is enough.
            # Averaging all descendants pulled every split child back to the
            # same XYZ and made evidence fight surface coverage.
            unique_ids, inverse = torch.unique(
                ids[matched], sorted=False, return_inverse=True
            )
            grouped_loss = torch.full(
                (len(unique_ids),),
                float("inf"),
                device=device,
                dtype=per_primitive.dtype,
            )
            grouped_loss.scatter_reduce_(
                0, inverse, per_primitive, reduce="amin", include_self=True
            )
            total = total + grouped_loss.sum()
            total_weight = total_weight + len(unique_ids)
            matched_total += int(len(unique_ids))
        if matched_total:
            self.consumed["track_factor"] += 1
        return total / total_weight.clamp_min(1), {
            "matched": matched_total,
            "sampled": sampled_total,
            "coverage": {
                str(source): sampler.audit()
                for source, sampler in self._track_samplers.items()
            },
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
            # Rotate the render-side responsibilities as well; unlike a fixed
            # stride this cannot starve the tail of an evolving topology.
            offset = (
                self.consumed["structure_factor"] * maximum_points
            ) % len(owned)
            owned = torch.roll(owned, shifts=-int(offset))[:maximum_points]
        points = surface.get_xyz[owned]
        unit_numpy = self._structure_sampler.next(maximum_points)
        unit_indices = torch.from_numpy(unit_numpy).to(
            device=device, dtype=torch.long
        )
        sampled_centers = structure["centers"][unit_indices]
        # Both sides rotate through complete epochs. This keeps the cdist
        # bounded while ensuring every permanent structural unit is consumed.
        distance = torch.cdist(points, sampled_centers)
        nearest_distance, nearest = distance.min(dim=1)
        nearest = unit_indices[nearest]
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
            "coverage": self._structure_sampler.audit(),
        }

    def coverage_audit(self) -> dict:
        return {
            "tracks": {
                str(source): sampler.audit()
                for source, sampler in self._track_samplers.items()
            },
            "structure": (
                None
                if self._structure_sampler is None
                else self._structure_sampler.audit()
            ),
        }

    def assert_complete_coverage(self) -> None:
        incomplete = []
        for name, audit in self.coverage_audit()["tracks"].items():
            if audit["never_visited"]:
                incomplete.append(f"track source {name}: {audit}")
        structure = self.coverage_audit()["structure"]
        if structure is not None and structure["never_visited"]:
            incomplete.append(f"structure: {structure}")
        if incomplete:
            raise RuntimeError(
                "Geometry evidence epoch did not reach full coverage: "
                + "; ".join(incomplete)
            )

    @property
    def geometry_view_stems(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.frame_by_stem)
                | set(self.pointmap_records)
                | set(self.moge3_records)
                | set(self.dav2_records)
            )
        )

    @property
    def metric_view_stems(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.frame_by_stem)
                | set(self.pointmap_records)
                | set(self.moge3_records)
            )
        )

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
            precision = self.chart.get(
                "precision",
                np.ones_like(self.chart["confidence"], dtype=np.float32),
            )[frame]
            valid = (
                self.chart["reference_mask"][frame]
                & np.isfinite(depth)
                & (depth > 0)
                & np.isfinite(confidence)
                & (confidence > 0)
                & np.isfinite(precision)
                & (precision > 0)
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
                (
                    valid
                    * np.clip(confidence, 0, 4)
                    * np.clip(precision, 0, 1)
                ).astype(np.float32),
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
        moge3_record = self.moge3_records.get(stem)
        if moge3_record is not None and self.moge3_runtime_enabled:
            if self.moge3_metric_to_cambridge_scale is None:
                raise AssertionError("Enabled MoGe3 runtime lacks its scene scale")
            if self.moge3_runtime_cache is not None:
                position = self.moge3_runtime_cache["camera_index"].get(stem)
                if position is None:
                    raise RuntimeError(
                        f"MoGe3 runtime cache lacks camera {stem}"
                    )
                moge3 = {
                    name: value[position]
                    for name, value in self.moge3_runtime_cache[
                        "arrays"
                    ].items()
                }
                valid = moge3["valid_mask"].astype(bool)
            else:
                moge3_path = Path(moge3_record["path"])
                if stem not in self._verified_moge3_stems:
                    if not moge3_path.is_file():
                        raise FileNotFoundError(moge3_path)
                    if moge3_path.stat().st_size != int(
                        moge3_record["bytes"]
                    ):
                        raise RuntimeError(
                            f"MoGe3 evidence size changed: {moge3_path}"
                        )
                    if moge3_sha256_file(moge3_path) != moge3_record["sha256"]:
                        raise RuntimeError(
                            f"MoGe3 evidence content changed: {moge3_path}"
                        )
                    self._verified_moge3_stems.add(stem)
                moge3 = self._moge3_cache.pop(stem, None)
                if moge3 is None:
                    complete_view = load_moge3_view(moge3_path)
                    retained_fields = {
                        "depth_m",
                        "normal_direct_camera",
                        "normal_depth_exact_k_camera",
                        "valid_mask",
                        "depth_normal_valid_mask",
                        "refinement_valid_mask",
                        "refinement_log_depth_std",
                        "refinement_final_delta_log_depth",
                    }
                    moge3 = {
                        name: complete_view[name]
                        for name in retained_fields
                    }
                self._moge3_cache[stem] = moge3
                while len(self._moge3_cache) > int(
                    self._moge3_cache_maximum_views
                ):
                    self._moge3_cache.popitem(last=False)
                valid = (
                    moge3["valid_mask"].astype(bool)
                    & moge3["refinement_valid_mask"].astype(bool)
                )
            result["moge3_depth_m"] = self._tensor(
                moge3["depth_m"]
                * float(self.moge3_metric_to_cambridge_scale),
                device=device,
                shape=shape,
            )
            result["moge3_valid_mask"] = self._tensor(
                valid.astype(np.float32),
                device=device,
                shape=shape,
                mode="nearest",
            )
            result["moge3_depth_normal_valid_mask"] = self._tensor(
                (
                    valid
                    & moge3["depth_normal_valid_mask"].astype(bool)
                ).astype(np.float32),
                device=device,
                shape=shape,
                mode="nearest",
            )
            result["moge3_refinement_log_depth_std"] = self._tensor(
                moge3["refinement_log_depth_std"],
                device=device,
                shape=shape,
            )
            result["moge3_refinement_final_delta_log_depth"] = self._tensor(
                moge3["refinement_final_delta_log_depth"],
                device=device,
                shape=shape,
            )
            direct_normal = self._tensor(
                moge3["normal_direct_camera"],
                device=device,
                shape=shape,
            )
            depth_normal = self._tensor(
                moge3["normal_depth_exact_k_camera"],
                device=device,
                shape=shape,
            )
            result["moge3_normal_direct_camera"] = F.normalize(
                direct_normal, dim=0, eps=1e-6
            )
            result["moge3_normal_depth_exact_k_camera"] = F.normalize(
                depth_normal, dim=0, eps=1e-6
            )
            self.consumed["moge3"] += 1
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
        nonempty_track_observation_cameras = sum(
            bool(len(record["pixels"]))
            for record in self.track_observations.values()
        )
        pointmap_posterior = getattr(
            self, "_pointmap_posterior_stats", {}
        )
        posterior_pixels = int(pointmap_posterior.get("pixels", 0))
        return {
            "available_geometry_views": len(self.frame_by_stem),
            "chart_base_source": self.chart_base_source,
            "active_metric_chart_views": (
                int(self.chart["active"].sum())
                if self.chart is not None
                else 0
            ),
            "available_dav2_views": len(self.dav2_records),
            "available_moge3_views": len(self.moge3_records),
            "moge3_runtime_enabled": bool(self.moge3_runtime_enabled),
            "moge3_compact_runtime_cache": bool(
                self.moge3_runtime_cache is not None
            ),
            "moge3_metric_to_cambridge_scale": (
                self.moge3_metric_to_cambridge_scale
            ),
            "moge3_chart_scale_available": bool(
                self.moge3_chart_scale_audit
            ),
            "moge3_chart_metric_to_cambridge_scale": (
                self.moge3_chart_scale_audit.get(
                    "metric_to_cambridge_scale"
                )
            ),
            "track_observation_camera_count": len(
                self.track_observations
            ),
            "nonempty_track_observation_camera_count": int(
                nonempty_track_observation_cameras
            ),
            "track_graph_representation": self.track_graph_contract[
                "representation"
            ],
            "descriptor_reciprocal_union_find_tracks": (
                self.track_graph_contract[
                    "descriptor_reciprocal_union_find_tracks"
                ]
            ),
            "track_graph_database_scope": self.track_graph_contract[
                "camera_scope"
            ],
            "source_consumption_count": dict(self.consumed),
            "source_losses_are_mutually_exclusive": True,
            "pointmap_cross_sequence_posterior": {
                "pointmap_camera_count": len(self.pointmap_records),
                "available_camera_count": int(
                    sum(
                        "cross_sequence_posterior" in record
                        for record in self.pointmap_records.values()
                    )
                ),
                "missing_camera_count": int(
                    sum(
                        "cross_sequence_posterior" not in record
                        for record in self.pointmap_records.values()
                    )
                ),
                "factor_calls": int(
                    pointmap_posterior.get("factor_calls", 0)
                ),
                "posterior_factor_calls": int(
                    pointmap_posterior.get("posterior_factor_calls", 0)
                ),
                "single_sequence_fallback_factor_calls": int(
                    pointmap_posterior.get(
                        "single_sequence_fallback_factor_calls", 0
                    )
                ),
                "pixels": posterior_pixels,
                "supported_pixels": int(
                    pointmap_posterior.get(
                        "cross_sequence_supported_pixels", 0
                    )
                ),
                "single_sequence_low_precision_pixels": int(
                    pointmap_posterior.get(
                        "single_sequence_low_precision_pixels", 0
                    )
                ),
                "missing_posterior_fallback_pixels": int(
                    pointmap_posterior.get(
                        "missing_posterior_fallback_pixels", 0
                    )
                ),
                "mean_precision": (
                    float(
                        pointmap_posterior.get("precision_sum", 0.0)
                    )
                    / posterior_pixels
                    if posterior_pixels
                    else 0.0
                ),
                "single_sequence_precision": (
                    SINGLE_SEQUENCE_POINTMAP_PRECISION
                ),
                "single_sequence_is_low_precision_not_deleted": True,
                "support_is_read_from_posterior_not_precision_threshold": (
                    True
                ),
            },
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
            "evidence_epoch_coverage": self.coverage_audit(),
        }
