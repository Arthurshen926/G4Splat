from __future__ import annotations

import json
import io
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def tensor_to_resized_mask(mask: torch.Tensor, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    if mask.ndim != 2:
        raise RuntimeError(f"Expected a 2D mask, got shape {tuple(mask.shape)}")
    height, width = shape
    mask = mask.to(device=device, dtype=torch.float32)[None, None]
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="nearest")
    return mask[0, 0] > 0.5


def tensor_to_resized_weight(
    weight: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise RuntimeError(f"Expected a 2D weight map, got shape {tuple(weight.shape)}")
    weight = weight.to(device=device, dtype=torch.float32)[None, None]
    if weight.shape[-2:] != shape:
        weight = F.interpolate(weight, size=shape, mode="bilinear", align_corners=False)
    return weight[0, 0].clamp(0.0, 1.0)


class _CpuTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda storage_bytes: torch.load(io.BytesIO(storage_bytes), map_location=torch.device("cpu"))
        return super().find_class(module, name)


def load_mask_dict(mask_pickle: Path) -> dict:
    with Path(mask_pickle).open("rb") as handle:
        masks = _CpuTorchUnpickler(handle).load()
    if not isinstance(masks, dict):
        raise RuntimeError(f"Expected {mask_pickle} to contain a dict, got {type(masks)}")
    return masks


def combine_masks(mask_tuple: tuple, mask_indices: list[int], shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    selected = []
    for index in mask_indices:
        if index < 0 or index >= len(mask_tuple):
            raise RuntimeError(f"Mask index {index} is outside tuple length {len(mask_tuple)}")
        selected.append(tensor_to_resized_mask(mask_tuple[index], shape, device))
    keep_mask = selected[0]
    for mask in selected[1:]:
        keep_mask = keep_mask & mask
    return keep_mask


def stack_masks_for_image_names(
    mask_lookup: "CambridgeMaskLookup",
    image_names: list[str],
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([
        mask_lookup.get_mask(image_name, shape, device)
        for image_name in image_names
    ])


def combine_optional_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    valid_masks = [mask.to(dtype=torch.bool) for mask in masks if mask is not None]
    if not valid_masks:
        return None

    combined = valid_masks[0]
    for mask in valid_masks[1:]:
        if mask.shape != combined.shape:
            raise RuntimeError(f"Mask shape {tuple(mask.shape)} does not match {tuple(combined.shape)}")
        combined = combined & mask
    return combined


def mask_to_weight(mask: torch.Tensor, invalid_weight: float = 0.0, feather_pixels: int = 0) -> torch.Tensor:
    if invalid_weight < 0.0 or invalid_weight > 1.0:
        raise ValueError(f"invalid_weight must be in [0, 1], got {invalid_weight}")
    if mask.ndim < 2:
        raise RuntimeError(f"Expected mask with at least 2 dimensions, got shape {tuple(mask.shape)}")

    keep_mask = mask.to(dtype=torch.bool)
    weights = torch.full_like(keep_mask, float(invalid_weight), dtype=torch.float32)
    weights = torch.where(keep_mask, torch.ones_like(weights), weights)

    if feather_pixels <= 0 or invalid_weight >= 1.0:
        return weights

    height, width = keep_mask.shape[-2:]
    flat_keep = keep_mask.reshape(-1, 1, height, width).to(dtype=torch.float32)
    flat_weights = weights.reshape(-1, height, width)
    assigned = keep_mask.reshape(-1, height, width).clone()
    frontier = flat_keep

    for ring_index in range(1, feather_pixels + 1):
        frontier = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1)
        ring = (frontier[:, 0] > 0.5) & (~assigned)
        ring_scale = (feather_pixels - ring_index + 1) / (feather_pixels + 1)
        ring_weight = invalid_weight + (1.0 - invalid_weight) * ring_scale
        flat_weights = torch.where(ring, torch.full_like(flat_weights, ring_weight), flat_weights)
        assigned = assigned | ring

    return flat_weights.reshape_as(weights)


class CambridgeMaskLookup:
    def __init__(self, dataset_path: Path, mask_pickle: Path, mask_indices: Optional[list[int]] = None):
        self.dataset_path = Path(dataset_path)
        self.mask_pickle = Path(mask_pickle)
        self.mask_indices = list(mask_indices or [0, 2])
        self.masks = load_mask_dict(self.mask_pickle)

        mapping_path = self.dataset_path / "name_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Expected {mapping_path} to map staged image names to source image names")
        self.staged_to_source = json.loads(mapping_path.read_text())
        # A Cambridge mask dictionary is encountered in two legitimate forms:
        # the original ULF/STDLoc pickle is keyed by nested source names
        # (``seq1/frame00001.png``), while the external-control adapter rekeys
        # its intentionally reduced pickle by staged COLMAP names
        # (``seq1__frame00001.png``).  Keep both sides of the mapping so the
        # lookup resolves to a *real pickle key*, rather than assuming the
        # mapping value is always the key.
        self.staged_by_stem = {
            Path(staged_name).stem: staged_name
            for staged_name in self.staged_to_source
        }
        if len(self.staged_by_stem) != len(self.staged_to_source):
            raise RuntimeError(
                "Staged Cambridge image names are ambiguous after extension "
                f"normalization in {mapping_path}"
            )
        self.source_by_stem = {
            Path(staged_name).stem: source_name for staged_name, source_name in self.staged_to_source.items()
        }
        self._valid_ratio_cache: dict[tuple[str, tuple[int, ...]], float] = {}
        self._resized_mask_cache: dict[tuple[str, int, int, str], torch.Tensor] = {}
        # Stacking the four 360x640 semantic masks used to be repeated for
        # every full-resolution training sample.  Keep the compact CPU stack;
        # this is roughly the same size as the source pickle and avoids four
        # independent host-to-device copies and interpolation launches.
        self._index_stack_cache: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}

    def with_indices(self, mask_indices: list[int]) -> "CambridgeMaskLookup":
        lookup = object.__new__(type(self))
        lookup.dataset_path = self.dataset_path
        lookup.mask_pickle = self.mask_pickle
        lookup.mask_indices = list(mask_indices)
        lookup.masks = self.masks
        lookup.staged_to_source = self.staged_to_source
        lookup.staged_by_stem = self.staged_by_stem
        lookup.source_by_stem = self.source_by_stem
        lookup._valid_ratio_cache = {}
        lookup._resized_mask_cache = {}
        lookup._index_stack_cache = {}
        return lookup

    def source_name_for(self, image_name: str) -> str:
        """Return the actual key of ``masks`` for a loaded camera name.

        The historical name of this method is retained because downstream
        support-map code uses it too, but its contract is intentionally the
        concrete mask key.  Returning a nested source name that is absent from
        an adapter's staged-key pickle deferred the error until a later direct
        ``lookup.masks[...]`` access and made the ULF-compatible control
        unusable.
        """
        if image_name in self.masks:
            return image_name

        image_basename = Path(image_name).name
        if image_basename in self.masks:
            return image_basename

        image_stem = Path(image_basename).stem
        staged_name = self.staged_by_stem.get(image_stem)
        if staged_name is not None and staged_name in self.masks:
            return staged_name

        if image_basename in self.staged_to_source:
            source_name = self.staged_to_source[image_basename]
            if source_name in self.masks:
                return source_name

        if image_stem in self.source_by_stem:
            source_name = self.source_by_stem[image_stem]
            if source_name in self.masks:
                return source_name

        # Staged Cambridge names are losslessly encoded as
        # ``sequence__frame``.  A QC subset's name_mapping.json may omit a
        # dense-training image even though the global mask pickle contains it.
        # Decode that canonical form instead of making mask lookup depend on
        # which dataset subset happened to provide the mapping file.
        if "__" in image_stem:
            sequence, frame = image_stem.split("__", 1)
            canonical_name = f"{sequence}/{frame}.png"
            if canonical_name in self.masks:
                return canonical_name

        raise RuntimeError(
            f"Could not resolve a mask key for staged image name {image_name!r} "
            f"using {self.dataset_path / 'name_mapping.json'} and "
            f"{self.mask_pickle}"
        )

    def get_mask(self, image_name: str, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        source_name = self.source_name_for(image_name)
        if source_name not in self.masks:
            raise RuntimeError(f"{source_name} from {self.dataset_path} not found in {self.mask_pickle}")
        height, width = shape
        target_device = torch.device(device)
        cache_key = (source_name, height, width, str(target_device))
        if cache_key not in self._resized_mask_cache:
            self._resized_mask_cache[cache_key] = combine_masks(
                self.masks[source_name],
                self.mask_indices,
                (height, width),
                target_device,
            )
        return self._resized_mask_cache[cache_key]

    def get_index_mask(
        self,
        image_name: str,
        index: int,
        shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        source_name = self.source_name_for(image_name)
        mask_tuple = self.masks[source_name]
        if index < 0 or index >= len(mask_tuple):
            raise RuntimeError(f"Mask index {index} is outside tuple length {len(mask_tuple)}")
        return tensor_to_resized_mask(mask_tuple[index], shape, device)

    def get_index_masks(
        self,
        image_name: str,
        indices: list[int] | tuple[int, ...],
        shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Return several keep masks through one transfer/interpolation.

        This is bit-equivalent to calling :meth:`get_index_mask` for each
        channel because nearest-neighbour interpolation is channel-separable.
        """
        source_name = self.source_name_for(image_name)
        mask_tuple = self.masks[source_name]
        selected = tuple(int(index) for index in indices)
        if not selected:
            raise RuntimeError("At least one mask index is required")
        for index in selected:
            if index < 0 or index >= len(mask_tuple):
                raise RuntimeError(
                    f"Mask index {index} is outside tuple length "
                    f"{len(mask_tuple)}"
                )
        cache_key = (source_name, selected)
        compact = self._index_stack_cache.get(cache_key)
        if compact is None:
            compact = torch.stack(
                [mask_tuple[index].to(dtype=torch.bool) for index in selected]
            ).contiguous()
            self._index_stack_cache[cache_key] = compact
        target = compact.to(device=device, dtype=torch.float32)[None]
        if target.shape[-2:] != tuple(shape):
            target = F.interpolate(target, size=shape, mode="nearest")
        return target[0] > 0.5

    def valid_ratio(self, image_name: str) -> float:
        return self.valid_ratio_for_indices(image_name, self.mask_indices)

    def valid_ratio_for_indices(self, image_name: str, mask_indices: list[int]) -> float:
        source_name = self.source_name_for(image_name)
        if source_name not in self.masks:
            raise RuntimeError(f"{source_name} from {self.dataset_path} not found in {self.mask_pickle}")

        indices = tuple(mask_indices)
        if not indices:
            raise RuntimeError("At least one mask index is required")
        cache_key = (source_name, indices)
        if cache_key in self._valid_ratio_cache:
            return self._valid_ratio_cache[cache_key]

        mask_tuple = self.masks[source_name]
        keep_mask: Optional[np.ndarray] = None
        for index in indices:
            if index < 0 or index >= len(mask_tuple):
                raise RuntimeError(f"Mask index {index} is outside tuple length {len(mask_tuple)}")
            mask = mask_tuple[index].detach().to(device="cpu", dtype=torch.bool).numpy()
            if keep_mask is None:
                keep_mask = mask.copy()
            else:
                if mask.shape != keep_mask.shape:
                    raise RuntimeError(f"Mask shape {tuple(mask.shape)} does not match {tuple(keep_mask.shape)}")
                np.logical_and(keep_mask, mask, out=keep_mask)

        ratio = float(np.count_nonzero(keep_mask) / keep_mask.size)
        self._valid_ratio_cache[cache_key] = ratio
        return ratio

    def invalid_ratio_for_indices(self, image_name: str, mask_indices: list[int]) -> float:
        return 1.0 - self.valid_ratio_for_indices(image_name, mask_indices)


class CambridgeTreeWeightLookup:
    """Build continuous canonical-tree weights from semantic masks and support maps."""

    def __init__(
        self,
        mask_lookup: CambridgeMaskLookup,
        *,
        tree_mask_index: int = 3,
        support_dir: Optional[Path] = None,
        rgb_floor: float = 0.25,
        rgb_support_gain: float = 0.50,
        geometry_floor: float = 0.05,
        geometry_support_gain: float = 0.25,
        planar_tree_weight: float = 0.0,
        sky_feather_pixels: int = 4,
        tree_feather_pixels: int = 6,
        missing_support_policy: str = "legacy_zero",
        neutral_support_value: float = 0.5,
    ):
        self.mask_lookup = mask_lookup
        self.tree_mask_index = int(tree_mask_index)
        self.support_dir = Path(support_dir) if support_dir is not None else None
        self.rgb_floor = float(rgb_floor)
        self.rgb_support_gain = float(rgb_support_gain)
        self.geometry_floor = float(geometry_floor)
        self.geometry_support_gain = float(geometry_support_gain)
        self.planar_tree_weight = float(planar_tree_weight)
        self.sky_feather_pixels = int(sky_feather_pixels)
        self.tree_feather_pixels = int(tree_feather_pixels)
        self.missing_support_policy = str(missing_support_policy)
        self.neutral_support_value = float(neutral_support_value)
        self._support_cache: dict[str, torch.Tensor] = {}
        self._support_valid_cache: dict[str, bool] = {}

        for name, value in (
            ("rgb_floor", self.rgb_floor),
            ("rgb_support_gain", self.rgb_support_gain),
            ("geometry_floor", self.geometry_floor),
            ("geometry_support_gain", self.geometry_support_gain),
            ("planar_tree_weight", self.planar_tree_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.missing_support_policy not in {"error", "neutral", "legacy_zero"}:
            raise ValueError(
                "missing_support_policy must be 'error', 'neutral', or 'legacy_zero'"
            )
        if not 0.0 <= self.neutral_support_value <= 1.0:
            raise ValueError("neutral_support_value must be in [0, 1]")

    def _support_path(self, image_name: str) -> Optional[Path]:
        if self.support_dir is None:
            return None
        source_name = self.mask_lookup.source_name_for(image_name)
        candidates = [
            self.support_dir / f"{Path(image_name).stem}.npy",
            self.support_dir / f"{source_name.replace('/', '__').rsplit('.', 1)[0]}.npy",
        ]
        return next((path for path in candidates if path.exists()), None)

    def support_coverage_audit(self, image_names: list[str]) -> dict:
        """Describe whether canonical support evidence exists for every view.

        Missing maps historically fall back to an all-zero support tensor.  That
        preserves the conservative tree-floor behavior, but it is *not*
        evidence that the tree pixels are non-canonical.  Report the distinction
        before training so a chart-only support directory cannot be mislabeled
        as full all-train canonical support.  File existence is insufficient:
        an exported all-zero map has exactly the same numerical effect as a
        missing map, so audit its values as well.
        """
        names = list(dict.fromkeys(str(name) for name in image_names))
        present = []
        missing = []
        stats_by_path: dict[Path, tuple[bool, float, float]] = {}
        for name in names:
            path = self._support_path(name)
            if path is None:
                missing.append(name)
            else:
                present.append(name)
                if path not in stats_by_path:
                    # mmap keeps this startup audit bounded even when a future
                    # all-train support export contains thousands of maps.
                    array = np.load(path, mmap_mode="r")
                    if array.ndim != 2 or not np.isfinite(array).all():
                        raise RuntimeError(f"Invalid tree support map: {path}")
                    clipped = np.clip(array, 0.0, 1.0)
                    stats_by_path[path] = (
                        bool(np.any(clipped > 0.0)),
                        float(np.mean(clipped)),
                        float(np.max(clipped)),
                    )
        requested = len(names)
        present_stats = [
            stats_by_path[self._support_path(name)]
            for name in present
        ]
        nonzero_map_count = sum(has_support for has_support, _, _ in present_stats)
        mean_support = (
            float(np.mean([mean for _, mean, _ in present_stats]))
            if present_stats
            else 0.0
        )
        max_support = (
            float(max(maximum for _, _, maximum in present_stats))
            if present_stats
            else 0.0
        )
        if not present:
            explicit_map_behavior = "no_explicit_support_maps"
        elif nonzero_map_count == 0:
            explicit_map_behavior = "all_explicit_maps_zero_support_tree_floor"
        elif nonzero_map_count < len(present):
            explicit_map_behavior = "some_explicit_maps_zero_support"
        else:
            explicit_map_behavior = "all_explicit_maps_have_nonzero_support"
        return {
            "support_dir": None if self.support_dir is None else str(self.support_dir),
            "requested_view_count": requested,
            "present_map_count": len(present),
            "missing_map_count": len(missing),
            "map_coverage_fraction": (
                float(len(present) / requested) if requested else 1.0
            ),
            "nonzero_map_count": nonzero_map_count,
            "zero_support_map_count": len(present) - nonzero_map_count,
            "nonzero_map_coverage_fraction": (
                float(nonzero_map_count / requested) if requested else 1.0
            ),
            "mean_support_value_across_present_maps": mean_support,
            "max_support_value_across_present_maps": max_support,
            "explicit_map_behavior": explicit_map_behavior,
            "missing_view_examples": missing[:20],
            "missing_map_behavior": (
                (
                    "neutral_prior_unknown_evidence"
                    if self.missing_support_policy == "neutral"
                    else "error_required_evidence"
                    if self.missing_support_policy == "error"
                    else "zero_support_tree_floor_not_canonical_evidence"
                )
                if missing else "all_requested_views_have_explicit_support_maps"
            ),
            "missing_support_policy": self.missing_support_policy,
            "neutral_support_value": self.neutral_support_value,
            "unknown_evidence_is_distinct_from_zero_support": True,
        }

    def support(self, image_name: str, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        source_name = self.mask_lookup.source_name_for(image_name)
        if source_name not in self._support_cache:
            path = self._support_path(image_name)
            if path is None:
                if self.missing_support_policy == "error":
                    raise FileNotFoundError(
                        "Tree support is required by policy but missing for "
                        f"{image_name} under {self.support_dir}"
                    )
                support = torch.full(
                    (1, 1),
                    self.neutral_support_value
                    if self.missing_support_policy == "neutral"
                    else 0.0,
                    dtype=torch.float32,
                )
                self._support_valid_cache[source_name] = False
            else:
                array = np.load(path)
                if array.ndim != 2 or not np.isfinite(array).all():
                    raise RuntimeError(f"Invalid tree support map: {path}")
                support = torch.from_numpy(array.astype(np.float32, copy=False)).clamp(0.0, 1.0)
                self._support_valid_cache[source_name] = True
            self._support_cache[source_name] = support
        return tensor_to_resized_weight(self._support_cache[source_name], shape, device)

    def support_valid(self, image_name: str) -> bool:
        """Whether the support value comes from an explicit evidence map."""
        source_name = self.mask_lookup.source_name_for(image_name)
        if source_name not in self._support_cache:
            # Populate the same cache path used by ``support`` without tying
            # validity to a particular raster resolution/device.
            self.support(image_name, (1, 1), torch.device("cpu"))
        return self._support_valid_cache[source_name]

    def components(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_keep = self.mask_lookup.get_mask(image_name, shape, device)
        base_weight = mask_to_weight(
            base_keep,
            invalid_weight=0.0,
            feather_pixels=self.sky_feather_pixels,
        )
        tree_keep = self.mask_lookup.get_index_mask(
            image_name,
            self.tree_mask_index,
            shape,
            device,
        )
        tree = (~tree_keep).to(dtype=torch.float32)
        if self.tree_feather_pixels > 0:
            kernel = 2 * self.tree_feather_pixels + 1
            tree = F.avg_pool2d(
                tree[None, None], kernel_size=kernel, stride=1,
                padding=self.tree_feather_pixels,
            )[0, 0]
        return base_weight, tree.clamp(0.0, 1.0), self.support(image_name, shape, device)

    def weights(
        self,
        image_name: str,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, tree, support = self.components(image_name, shape, device)
        rgb_tree = self.rgb_floor + self.rgb_support_gain * support
        geometry_tree = self.geometry_floor + self.geometry_support_gain * support
        rgb = base * ((1.0 - tree) + tree * rgb_tree)
        geometry = base * ((1.0 - tree) + tree * geometry_tree)
        planar = base * ((1.0 - tree) + tree * self.planar_tree_weight)
        return rgb.clamp(0.0, 1.0), geometry.clamp(0.0, 1.0), planar.clamp(0.0, 1.0)
