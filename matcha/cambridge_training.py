from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from matcha.dm_regularization.depth import compute_depth_order_loss


class PerImageAffineColorCorrection(torch.nn.Module):
    """Regularized exposure correction keyed by real training-image name."""

    def __init__(self, image_names: Iterable[str]):
        super().__init__()
        unique_names = list(dict.fromkeys(str(name) for name in image_names))
        if not unique_names:
            raise RuntimeError("Per-image color correction requires at least one image name")
        self.image_name_to_index = {
            image_name: index for index, image_name in enumerate(unique_names)
        }
        self.log_scales = torch.nn.Parameter(torch.zeros((len(unique_names), 3, 1, 1)))
        self.biases = torch.nn.Parameter(torch.zeros((len(unique_names), 3, 1, 1)))

    def forward(self, image: torch.Tensor, image_name: str) -> torch.Tensor:
        image_name = str(image_name)
        if image_name not in self.image_name_to_index:
            raise RuntimeError(f"Missing color correction parameters for image {image_name!r}")
        index = self.image_name_to_index[image_name]
        scale = torch.exp(self.log_scales[index]).to(device=image.device, dtype=image.dtype)
        bias = self.biases[index].to(device=image.device, dtype=image.dtype)
        return image * scale + bias

    def identity_regularization(self) -> torch.Tensor:
        return self.log_scales.square().mean() + self.biases.square().mean()


def prepare_loss_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise RuntimeError(f"Expected a 2D/3D mask, got {tuple(mask.shape)}")
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask[None], size=reference.shape[-2:], mode="nearest")[0]
    if mask.shape[0] not in (1, reference.shape[0]):
        raise RuntimeError(
            f"Mask channels {mask.shape[0]} do not match reference channels {reference.shape[0]}"
        )
    return mask


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if values.ndim == 2:
        weight = mask.squeeze().to(device=values.device, dtype=values.dtype)
        if weight.shape != values.shape:
            weight = F.interpolate(
                weight[None, None], size=values.shape, mode="nearest"
            )[0, 0]
        denominator = weight.sum()
        if denominator <= 0:
            return values.sum() * 0.0
        return (values * weight).sum() / denominator
    weight = prepare_loss_mask(mask, values)
    if weight.shape[0] == 1 and values.shape[0] != 1:
        weight = weight.expand(values.shape[0], -1, -1)
    denominator = weight.sum()
    if denominator <= 0:
        return values.sum() * 0.0
    return (values * weight).sum() / denominator


def photometric_difference(
    residual: torch.Tensor,
    loss_type: str = "l1",
    charbonnier_eps: float = 1e-3,
) -> torch.Tensor:
    if loss_type == "l1":
        return residual.abs()
    if loss_type == "charbonnier":
        return torch.sqrt(residual.square() + charbonnier_eps**2)
    raise ValueError(f"Unsupported RGB loss type: {loss_type}")


def compute_rgb_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    lambda_dssim: float,
    ssim_fn,
    loss_type: str = "l1",
    charbonnier_eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (
        torch.ones((1, *image.shape[-2:]), device=image.device, dtype=image.dtype)
        if mask is None
        else prepare_loss_mask(mask, image)
    )
    pixel_loss = photometric_difference(image - target, loss_type, charbonnier_eps)
    rgb_loss = masked_mean(pixel_loss, weight)
    if lambda_dssim <= 0.0:
        return rgb_loss, rgb_loss
    dssim = 1.0 - ssim_fn(image * weight, target * weight)
    if mask is not None:
        # Scalar SSIM averages the all-zero excluded region as a perfect match.
        # Renormalizing by valid area keeps a large sky mask from silently
        # diluting the RGB objective. Boundary windows remain conservative.
        valid_fraction = weight[0].mean()
        if valid_fraction <= 0:
            dssim = image.sum() * 0.0
        else:
            dssim = (dssim / valid_fraction).clamp(min=0.0, max=2.0)
    return rgb_loss, (1.0 - lambda_dssim) * rgb_loss + lambda_dssim * dssim


def compute_masked_depth_order_loss(
    *,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if mask is None:
        return compute_depth_order_loss(reduction="mean", **kwargs)
    loss_map = compute_depth_order_loss(reduction="none", **kwargs)
    return masked_mean(loss_map, mask)


def combine_boolean_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    present = [mask.to(dtype=torch.bool) for mask in masks if mask is not None]
    if not present:
        return None
    result = present[0]
    for mask in present[1:]:
        if mask.shape != result.shape:
            raise RuntimeError(f"Cannot combine masks {tuple(result.shape)} and {tuple(mask.shape)}")
        result = result & mask
    return result


def geometry_iteration(
    iteration: int,
    *,
    use_dense_supervision: bool,
    every_n: int,
    dense_only_from_iter: int,
) -> bool:
    if not use_dense_supervision:
        return True
    if dense_only_from_iter >= 0 and iteration > dense_only_from_iter:
        return False
    return iteration % max(int(every_n), 1) == 0


def rgb_supervision_weight(
    *,
    is_pseudo_view: bool,
    is_geometry_view: bool,
    use_dense_supervision: bool,
    downweight_input_view_color_loss: bool,
    pseudo_rgb_weight: float,
) -> float:
    """Return the RGB multiplier without weakening the only real observations."""
    should_downweight = is_pseudo_view or (
        is_geometry_view
        and use_dense_supervision
        and downweight_input_view_color_loss
    )
    return float(pseudo_rgb_weight if should_downweight else 1.0)


def dense_depth_weight(iteration: int, schedule: str) -> float:
    if schedule == "none":
        return 0.0
    if schedule == "strong":
        return 1.0
    if schedule == "strong_decay":
        if iteration <= 7000:
            return 1.0
        if iteration <= 15000:
            return 0.1
        if iteration <= 20000:
            return 0.01
        if iteration <= 25000:
            return 0.001
        return 0.0001
    if schedule == "weak":
        return 0.1 if iteration > 3000 else 0.0
    if schedule != "default":
        raise ValueError(f"Unknown dense regularization schedule: {schedule}")
    if iteration <= 3000:
        return 0.0
    if iteration <= 7000:
        return 1.0
    if iteration <= 15000:
        return 0.1
    if iteration <= 20000:
        return 0.01
    if iteration <= 25000:
        return 0.001
    return 0.0001


def linear_weight(iteration: int, start: float, end: float, decay_until: int) -> float:
    if decay_until <= 0:
        return float(end)
    ratio = min(max(iteration / decay_until, 0.0), 1.0)
    return float(start + ratio * (end - start))


def sanitize_depth(
    depth: torch.Tensor,
    *,
    confidence: Optional[torch.Tensor] = None,
    semantic_mask: Optional[torch.Tensor] = None,
    max_abs_depth: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = depth.squeeze()
    valid = torch.isfinite(depth) & (depth > 0)
    if max_abs_depth is not None:
        valid &= depth.abs() <= max_abs_depth
    if confidence is not None:
        confidence = confidence.squeeze().to(device=depth.device)
        if confidence.shape != depth.shape:
            confidence = F.interpolate(
                confidence[None, None].float(), size=depth.shape, mode="nearest"
            )[0, 0]
        valid &= confidence > 0.5
    if semantic_mask is not None:
        semantic_mask = semantic_mask.squeeze().to(device=depth.device, dtype=torch.bool)
        if semantic_mask.shape != depth.shape:
            semantic_mask = F.interpolate(
                semantic_mask[None, None].float(), size=depth.shape, mode="nearest"
            )[0, 0] > 0.5
        valid &= semantic_mask
    return torch.where(valid, depth, torch.zeros_like(depth)), valid


def sanitize_chart_geometry(
    charts_data: dict[str, torch.Tensor],
    *,
    max_abs_depth: Optional[float] = 50.0,
    max_abs_point: Optional[float] = 50.0,
    min_confidence: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Remove unsupported or extreme chart pixels before plane construction."""
    required = ("depths", "prior_depths", "pts", "confs")
    missing = [key for key in required if key not in charts_data]
    if missing:
        raise RuntimeError(f"Charts data is missing required keys: {missing}")

    depths = charts_data["depths"]
    prior_depths = charts_data["prior_depths"]
    points = charts_data["pts"]
    confidences = charts_data["confs"]
    if depths.shape != prior_depths.shape or depths.shape != confidences.shape:
        raise RuntimeError("Chart depth, prior depth, and confidence shapes must match")
    if points.shape[:-1] != depths.shape or points.shape[-1] != 3:
        raise RuntimeError("Chart point shape must be depth shape plus a 3D coordinate")

    valid = torch.isfinite(confidences) & (confidences > min_confidence)
    for depth in (depths, prior_depths):
        valid &= torch.isfinite(depth) & (depth > 0)
        if max_abs_depth is not None and max_abs_depth > 0:
            valid &= depth.abs() <= max_abs_depth
    valid &= torch.isfinite(points).all(dim=-1)
    if max_abs_point is not None and max_abs_point > 0:
        valid &= (points.abs() <= max_abs_point).all(dim=-1)

    sanitized = dict(charts_data)
    for key in ("depths", "prior_depths", "confs"):
        sanitized[key] = torch.where(valid, charts_data[key], torch.zeros_like(charts_data[key]))
    sanitized["pts"] = torch.where(
        valid[..., None],
        points,
        torch.zeros_like(points),
    )
    return sanitized, valid


def save_depth_cache(path: Path, image_names: Iterable[str], depths: Iterable[torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "image_names": list(image_names),
        "depths": [depth.detach().to(device="cpu", dtype=torch.float16) for depth in depths],
    }
    torch.save(payload, path)


def load_depth_cache(path: Path, expected_names: Iterable[str]) -> list[torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    expected_names = list(expected_names)
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported dense depth cache version in {path}")
    if payload.get("image_names") != expected_names:
        raise RuntimeError(f"Dense depth cache camera order does not match {path}")
    depths = payload.get("depths")
    if not isinstance(depths, list) or len(depths) != len(expected_names):
        raise RuntimeError(f"Malformed dense depth cache: {path}")
    return depths
