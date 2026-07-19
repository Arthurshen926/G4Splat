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
        self.source_by_stem = {
            Path(staged_name).stem: source_name for staged_name, source_name in self.staged_to_source.items()
        }
        self._valid_ratio_cache: dict[tuple[str, tuple[int, ...]], float] = {}
        self._resized_mask_cache: dict[tuple[str, int, int, str], torch.Tensor] = {}

    def with_indices(self, mask_indices: list[int]) -> "CambridgeMaskLookup":
        lookup = object.__new__(type(self))
        lookup.dataset_path = self.dataset_path
        lookup.mask_pickle = self.mask_pickle
        lookup.mask_indices = list(mask_indices)
        lookup.masks = self.masks
        lookup.staged_to_source = self.staged_to_source
        lookup.source_by_stem = self.source_by_stem
        lookup._valid_ratio_cache = {}
        lookup._resized_mask_cache = {}
        return lookup

    def source_name_for(self, image_name: str) -> str:
        if image_name in self.masks:
            return image_name

        image_basename = Path(image_name).name
        if image_basename in self.staged_to_source:
            return self.staged_to_source[image_basename]

        image_stem = Path(image_basename).stem
        if image_stem in self.source_by_stem:
            return self.source_by_stem[image_stem]

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

        raise RuntimeError(f"Could not map staged image name {image_name!r} using {self.dataset_path / 'name_mapping.json'}")

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
        self._support_cache: dict[str, torch.Tensor] = {}

        for name, value in (
            ("rgb_floor", self.rgb_floor),
            ("rgb_support_gain", self.rgb_support_gain),
            ("geometry_floor", self.geometry_floor),
            ("geometry_support_gain", self.geometry_support_gain),
            ("planar_tree_weight", self.planar_tree_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def _support_path(self, image_name: str) -> Optional[Path]:
        if self.support_dir is None:
            return None
        source_name = self.mask_lookup.source_name_for(image_name)
        candidates = [
            self.support_dir / f"{Path(image_name).stem}.npy",
            self.support_dir / f"{source_name.replace('/', '__').rsplit('.', 1)[0]}.npy",
        ]
        return next((path for path in candidates if path.exists()), None)

    def support(self, image_name: str, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        source_name = self.mask_lookup.source_name_for(image_name)
        if source_name not in self._support_cache:
            path = self._support_path(image_name)
            if path is None:
                support = torch.zeros((1, 1), dtype=torch.float32)
            else:
                array = np.load(path)
                if array.ndim != 2 or not np.isfinite(array).all():
                    raise RuntimeError(f"Invalid tree support map: {path}")
                support = torch.from_numpy(array.astype(np.float32, copy=False)).clamp(0.0, 1.0)
            self._support_cache[source_name] = support
        return tensor_to_resized_weight(self._support_cache[source_name], shape, device)

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
