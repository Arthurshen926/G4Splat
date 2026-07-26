"""Runtime semantic task fields for honest Cambridge residual refinement.

The available Cambridge pickle provides keep masks, not a dense semantic
segmentation.  Building and ground therefore remain explicit rigid proxies;
trunk remains unavailable.  The fields in this module are deterministic
runtime tensors and are consumed directly by loss and topology allocation.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from matcha.cambridge_masks import CambridgeMaskLookup


TASK_FIELD_VERSION = "cambridge_runtime_task_fields_v2"


def _soft_boundary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    value = mask.to(dtype=torch.float32)
    kernel = 2 * int(radius) + 1
    extrema = F.max_pool2d(
        torch.stack([value, -value])[:, None],
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return (extrema[0, 0] + extrema[1, 0]).clamp(0.0, 1.0)


def _scale_aware_radius(mask: torch.Tensor, base_radius: int) -> int:
    """Resolution/foreground-scale-aware fallback without invented depth."""
    if base_radius <= 0 or not bool(mask.any()):
        return 0
    height, width = mask.shape
    equivalent_radius = torch.sqrt(
        mask.float().sum() / torch.pi
    ).item()
    resolution_scale = ((height * width) / (540.0 * 960.0)) ** 0.25
    component_scale = max(0.5, min(2.0, (equivalent_radius / 120.0) ** 0.5))
    return int(round(max(1.0, min(8.0, base_radius * resolution_scale * component_scale))))


class OutdoorTaskFieldLookup:
    """Materialize per-pixel task probabilities from the four real mask channels."""

    def __init__(
        self,
        dataset_path: Path,
        tree_mask_pickle: Path,
        semantic_manifest: Path,
        *,
        boundary_radius: int = 3,
        max_cached_views: int = 16,
    ):
        self.dataset_path = Path(dataset_path).resolve()
        self.tree_mask_pickle = Path(tree_mask_pickle).resolve()
        self.semantic_manifest_path = Path(semantic_manifest).resolve()
        self.boundary_radius = int(boundary_radius)
        self.max_cached_views = int(max_cached_views)
        if self.boundary_radius < 0:
            raise ValueError("boundary_radius must be non-negative")
        if self.max_cached_views < 0:
            raise ValueError("max_cached_views must be non-negative")
        if not self.semantic_manifest_path.is_file():
            raise FileNotFoundError(self.semantic_manifest_path)
        self.semantic_manifest = json.loads(
            self.semantic_manifest_path.read_text(encoding="utf-8")
        )
        availability = self.semantic_manifest.get("class_availability", {})
        if availability.get("canopy") != "tree_mask_index_3":
            raise RuntimeError(
                "Residual task fields require a real tree mask at channel index 3"
            )
        self.lookup = CambridgeMaskLookup(
            self.dataset_path,
            self.tree_mask_pickle,
            mask_indices=[0, 1, 2, 3],
        )
        self._mask_cache: OrderedDict[
            tuple[str, int, int, str], tuple[torch.Tensor, ...]
        ] = OrderedDict()
        # The adaptive radius requires a scalar GPU reduction.  Cache the two
        # deterministic integers per camera/resolution so subsequent epochs
        # do not introduce four device synchronizations per training step.
        self._radius_cache: dict[
            tuple[str, int, int], tuple[int, int]
        ] = {}
        self._canopy_fraction_cache: dict[str, float] = {}

    def _keep_masks(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        key = (str(image_name), int(shape[0]), int(shape[1]), str(torch.device(device)))
        if key in self._mask_cache:
            masks = self._mask_cache.pop(key)
            self._mask_cache[key] = masks
            return masks
        stacked = self.lookup.get_index_masks(
            image_name,
            (0, 1, 2, 3),
            shape,
            device,
        )
        masks = tuple(stacked[index] for index in range(4))
        if self.max_cached_views:
            self._mask_cache[key] = masks
            while len(self._mask_cache) > self.max_cached_views:
                self._mask_cache.popitem(last=False)
        return masks

    def fields(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        object_keep, sky_keep, distortion_keep, tree_keep = self._keep_masks(
            image_name, shape, device
        )
        p_transient = (~object_keep).float()
        p_sky = (~sky_keep).float() * (1.0 - p_transient)
        p_canopy = (
            (~tree_keep).float() * (1.0 - p_transient) * (1.0 - p_sky)
        )
        p_rigid = (
            object_keep.float()
            * sky_keep.float()
            * distortion_keep.float()
            * tree_keep.float()
        )
        uncertain_union = (p_transient + p_sky + p_canopy).clamp(0.0, 1.0)
        radius_key = (str(image_name), int(shape[0]), int(shape[1]))
        cached_radii = self._radius_cache.get(radius_key)
        if cached_radii is None:
            adaptive_radius = _scale_aware_radius(
                uncertain_union > 0.5, self.boundary_radius
            )
            canopy_radius = _scale_aware_radius(
                p_canopy > 0.5, self.boundary_radius
            )
            cached_radii = (adaptive_radius, canopy_radius)
            self._radius_cache[radius_key] = cached_radii
        else:
            adaptive_radius, canopy_radius = cached_radii
        p_boundary = _soft_boundary(uncertain_union > 0.5, adaptive_radius)
        p_canopy_boundary = _soft_boundary(
            p_canopy > 0.5, canopy_radius
        ) * p_canopy
        p_canopy_core = p_canopy * (1.0 - p_canopy_boundary)
        distortion = distortion_keep.float()

        # Actual loss/task weights.  Building and ground are intentionally
        # aliases of the only available rigid proxy, not invented labels.
        w_gaussian_rgb = (
            distortion
            * (0.15 * p_rigid + 1.5 * p_canopy)
            * (1.0 - p_sky)
            * (1.0 - 0.8 * p_transient)
            * (1.0 - 0.5 * p_boundary)
        )
        w_sky_rgb = (
            distortion
            * p_sky
            * (1.0 - 0.8 * p_transient)
            * (1.0 - 0.35 * p_boundary)
        )
        w_topology = (
            distortion
            * p_canopy
            * (1.0 - p_transient)
            * (1.0 - p_sky)
            * (1.0 - p_boundary)
        )
        w_geometry = distortion * (p_rigid + 0.1 * p_canopy)
        w_plane = distortion * p_rigid * (1.0 - p_boundary)

        return {
            "p_building": p_rigid,
            "p_ground": p_rigid,
            "p_rigid": p_rigid,
            "p_trunk": torch.zeros_like(p_rigid),
            "p_branch": torch.zeros_like(p_rigid),
            "p_canopy": p_canopy,
            "p_canopy_core": p_canopy_core,
            "p_canopy_boundary": p_canopy_boundary,
            "p_sky": p_sky,
            "p_transient": p_transient,
            "p_boundary_uncertain": p_boundary,
            "w_rgb": distortion * (1.0 - 0.8 * p_transient),
            "w_gaussian_rgb": w_gaussian_rgb,
            "w_sky_rgb": w_sky_rgb,
            "w_geometry": w_geometry,
            "w_plane": w_plane,
            "w_matching": w_plane,
            "w_topology": w_topology,
        }

    def canopy_fraction(
        self,
        image_name: str,
        shape: tuple[int, int],
    ) -> float:
        """Fast source-grid CPU ranking signal.

        Ranking does not need a training-resolution tensor.  Avoid stacking
        and resizing all four channels for every camera at cold start; doing
        that for the 1,487-view Cambridge set used several CPU-minutes and
        retained gigabytes of compact-mask cache entries before iteration one.
        A bounded regular sample of the source grid is sufficient for this
        view-priority statistic.
        """
        source_name = self.lookup.source_name_for(image_name)
        cached = self._canopy_fraction_cache.get(source_name)
        if cached is not None:
            return cached
        mask_tuple = self.lookup.masks[source_name]
        source_height, source_width = mask_tuple[0].shape
        target_height = max(int(shape[0]), 1)
        target_width = max(int(shape[1]), 1)
        row_stride = max(
            (source_height + target_height - 1) // target_height, 1
        )
        column_stride = max(
            (source_width + target_width - 1) // target_width, 1
        )
        sample = (
            slice(None, None, row_stride),
            slice(None, None, column_stride),
        )
        object_keep = mask_tuple[0][sample].to(
            device="cpu", dtype=torch.bool
        )
        sky_keep = mask_tuple[1][sample].to(
            device="cpu", dtype=torch.bool
        )
        tree_keep = mask_tuple[3][sample].to(
            device="cpu", dtype=torch.bool
        )
        if not (
            object_keep.shape == sky_keep.shape == tree_keep.shape
        ):
            raise RuntimeError(
                f"Semantic mask shapes differ for {source_name}"
            )
        canopy = (~tree_keep) & object_keep & sky_keep
        fraction = float(
            canopy.sum(dtype=torch.float32).item() / canopy.numel()
        )
        self._canopy_fraction_cache[source_name] = fraction
        return fraction

    def audit(self) -> dict:
        return {
            "version": TASK_FIELD_VERSION,
            "materialization": "deterministic_runtime_per_sample",
            "source_mask_pickle": str(self.tree_mask_pickle),
            "semantic_manifest": str(self.semantic_manifest_path),
            "mask_channel_contract": {
                "transient": "inverse_keep_channel_0",
                "sky": "inverse_keep_channel_1",
                "distortion_valid": "keep_channel_2",
                "canopy": "inverse_keep_channel_3",
                "trunk": "unavailable_zero",
                "building": "rigid_proxy_not_semantic_segmentation",
                "ground": "rigid_proxy_not_semantic_segmentation",
            },
            "boundary_policy": {
                "type": "resolution_and_foreground_scale_aware_global_fallback",
                "base_radius_pixels_at_540x960": self.boundary_radius,
                "range_pixels": [1, 8],
                "limitation": (
                    "independent component depth is unavailable; no metric-depth "
                    "boundary scale is invented"
                ),
            },
            "max_cached_views": self.max_cached_views,
        }
