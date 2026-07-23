from __future__ import annotations

import json
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


def save_per_image_affine_color_correction(
    correction: PerImageAffineColorCorrection,
    path: str | Path,
) -> None:
    """Persist the train-camera affine model beside a Gaussian checkpoint.

    The affine parameters participate directly in the RGB objective.  Saving
    only the Gaussian PLY would otherwise make a later train-view render use
    a different image model than the optimization that produced the PLY.
    """
    image_names = [
        name
        for name, _ in sorted(
            correction.image_name_to_index.items(), key=lambda item: item[1]
        )
    ]
    state_dict = {
        name: value.detach().cpu()
        for name, value in correction.state_dict().items()
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "image_names": image_names,
            "state_dict": state_dict,
        },
        destination,
    )


def load_per_image_affine_color_correction(
    path: str | Path,
    *,
    device: torch.device | str,
) -> PerImageAffineColorCorrection:
    """Load a persisted affine model with its image-name contract intact."""
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise RuntimeError(f"Unsupported affine color-correction state: {path}")
    image_names = payload.get("image_names")
    state_dict = payload.get("state_dict")
    if not isinstance(image_names, list) or not image_names or not isinstance(state_dict, dict):
        raise RuntimeError(f"Malformed affine color-correction state: {path}")
    correction = PerImageAffineColorCorrection(image_names)
    correction.load_state_dict(state_dict, strict=True)
    return correction.to(device).eval()


def apply_per_image_affine_color_correction(
    correction: Optional[PerImageAffineColorCorrection],
    image: torch.Tensor,
    image_name: str,
) -> torch.Tensor:
    """Apply the learned camera correction, or identity for a novel camera."""
    if correction is None or str(image_name) not in correction.image_name_to_index:
        return image
    return correction(image, image_name)


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


def fused_inverse_depth_nll(
    rendered_depth: torch.Tensor,
    target_depth: torch.Tensor,
    rho_variance: torch.Tensor,
    *,
    confidence: Optional[torch.Tensor] = None,
    source_bitmask: Optional[torch.Tensor] = None,
    mono_only_weight: float = 0.15,
    variance_floor: float = 0.05,
    variance_ceiling: float = 20.0,
) -> torch.Tensor:
    """Variance-aware inverse-depth likelihood for fused outdoor geometry.

    The fusion stage stores variance in inverse-depth space.  Its absolute
    calibration can differ between cameras, so normalize it by the robust
    median for the current target map before evaluating the likelihood.  This
    preserves its relative certainty while keeping one unusually large-scale
    Chart from changing the global loss scale.  Pixels backed only by a mono
    source receive a deliberately weak weight; plane/Chart evidence remains
    the primary geometric anchor.
    """
    if not 0.0 <= mono_only_weight <= 1.0:
        raise ValueError("mono_only_weight must be in [0, 1]")
    if not 0.0 < variance_floor <= variance_ceiling:
        raise ValueError("variance bounds must satisfy 0 < floor <= ceiling")

    def single_map(value: torch.Tensor, *, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = value.to(device=rendered_depth.device, dtype=dtype)
        if result.ndim == 2:
            return result
        if result.ndim == 3 and result.shape[0] == 1:
            return result[0]
        if result.ndim == 4 and result.shape[:2] == (1, 1):
            return result[0, 0]
        raise RuntimeError(
            f"{name} must be a single [height,width] depth map, got {tuple(result.shape)}"
        )

    # Gaussian renderer depth is [1,H,W], whereas persisted fusion maps are
    # [H,W].  Canonicalize the per-view loss to 2D before boolean indexing so
    # variance/provenance maps cannot be accidentally broadcast to a new axis.
    rendered_depth = single_map(rendered_depth, name="rendered_depth", dtype=rendered_depth.dtype)
    target_depth = single_map(target_depth, name="target_depth", dtype=rendered_depth.dtype)
    variance = single_map(rho_variance, name="rho_variance", dtype=rendered_depth.dtype)
    if target_depth.shape != rendered_depth.shape:
        target_depth = F.interpolate(
            target_depth.reshape(1, 1, *target_depth.shape[-2:]),
            size=rendered_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    if variance.shape != rendered_depth.shape:
        variance = F.interpolate(
            variance.reshape(1, 1, *variance.shape[-2:]),
            size=rendered_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    valid = (
        torch.isfinite(rendered_depth)
        & torch.isfinite(target_depth)
        & torch.isfinite(variance)
        & (rendered_depth > 1e-8)
        & (target_depth > 1e-8)
        & (variance > 0.0)
    )
    if not torch.any(valid):
        return rendered_depth.sum() * 0.0
    reference_variance = variance[valid].detach().median().clamp_min(1e-12)
    normalized_variance = (variance / reference_variance).clamp(
        min=variance_floor,
        max=variance_ceiling,
    )
    rendered_rho = torch.reciprocal(rendered_depth.clamp_min(1e-8))
    target_rho = torch.reciprocal(target_depth.clamp_min(1e-8))
    nll = 0.5 * (
        (rendered_rho - target_rho).square() / normalized_variance
        + torch.log(normalized_variance)
    )
    weight = valid.to(dtype=rendered_depth.dtype)
    if confidence is not None:
        confidence = single_map(confidence, name="confidence", dtype=rendered_depth.dtype)
        if confidence.shape != rendered_depth.shape:
            confidence = F.interpolate(
                confidence.reshape(1, 1, *confidence.shape[-2:]),
                size=rendered_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        weight = weight * confidence.clamp(0.0, 1.0)
    if source_bitmask is not None:
        source_bitmask = single_map(source_bitmask, name="source_bitmask", dtype=torch.uint8)
        if source_bitmask.shape != rendered_depth.shape:
            source_bitmask = F.interpolate(
                source_bitmask.reshape(1, 1, *source_bitmask.shape[-2:]).to(torch.float32),
                size=rendered_depth.shape[-2:],
                mode="nearest",
            )[0, 0].to(torch.uint8)
        multiview = (source_bitmask & 0b0011) != 0
        mono_only = ((source_bitmask & 0b0100) != 0) & (~multiview)
        source_weight = torch.where(
            multiview,
            torch.ones_like(weight),
            torch.where(mono_only, torch.full_like(weight, mono_only_weight), torch.zeros_like(weight)),
        )
        weight = weight * source_weight
    return masked_mean(nll, weight)


def build_spatial_camera_blocks(
    camera_centers: torch.Tensor,
    *,
    bins: int = 4,
) -> list[list[int]]:
    """Partition real cameras into deterministic, non-empty spatial blocks.

    The blocks are used for sampling rather than for a hard geometric split:
    each dense-training cycle draws one camera from every occupied block.  It
    prevents a long, high-frame-rate traversal from monopolizing RGB and
    topology statistics while keeping all real views eligible.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    centers = torch.as_tensor(camera_centers, dtype=torch.float32).detach().cpu()
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"camera_centers must have shape [N, 3], got {tuple(centers.shape)}")
    if centers.shape[0] == 0:
        raise ValueError("at least one camera center is required")
    lower = torch.quantile(centers, 0.02, dim=0)
    upper = torch.quantile(centers, 0.98, dim=0)
    span = (upper - lower).clamp_min(1e-6)
    coordinates = torch.floor((centers - lower) / span * bins).to(torch.int64)
    coordinates = coordinates.clamp(min=0, max=bins - 1)
    codes = coordinates[:, 0] * bins * bins + coordinates[:, 1] * bins + coordinates[:, 2]
    blocks: list[list[int]] = []
    for code in torch.unique(codes, sorted=True).tolist():
        blocks.append(torch.nonzero(codes == code, as_tuple=False).flatten().tolist())
    return blocks


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


def ulfloc_masked_supervision(
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    object_mask: torch.Tensor,
    sky_mask: torch.Tensor,
    distortion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce ULF-Loc's legacy Cambridge RGB pixel protocol exactly.

    ULF-Loc does not simply discard every semantic-invalid pixel.  It masks
    the rendered/target image by ``object & distortion`` and then writes a
    white target where the sky channel is false.  Keeping this slightly odd
    ordering explicit is useful for an honest standard-protocol control: a
    generic AND mask is a *different* training objective.

    The returned mask is the object/distortion keep mask and is also what the
    original 2DGS branch applied to its normal and distortion maps.
    """
    if image.shape != target.shape:
        raise RuntimeError(
            f"Image/target shape mismatch: {tuple(image.shape)} vs {tuple(target.shape)}"
        )
    object_keep = prepare_loss_mask(object_mask, image)[0] > 0.5
    sky_keep = prepare_loss_mask(sky_mask, image)[0] > 0.5
    distortion_keep = prepare_loss_mask(distortion_mask, image)[0] > 0.5
    rgb_keep = object_keep & distortion_keep
    weight = rgb_keep.to(dtype=image.dtype)[None]
    masked_image = image * weight
    masked_target = target * weight
    # This is deliberately after object/distortion masking, matching the
    # upstream ULF-Loc code path rather than a more conventional semantic AND.
    masked_target = torch.where(
        sky_keep[None], masked_target, torch.ones_like(masked_target)
    )
    return masked_image, masked_target, rgb_keep


def compute_masked_depth_order_loss(
    *,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if mask is None:
        return compute_depth_order_loss(reduction="mean", **kwargs)
    # The order residual compares a pixel with a randomly shifted neighbour.
    # Suppressing only the source pixel still lets invalid sky/tree/hole
    # neighbours exert a geometric force.  Delegate the paired-support
    # normalization to the primitive so both endpoints must be valid.
    return compute_depth_order_loss(reduction="mean", valid_mask=mask, **kwargs)


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
    geometry_schedule: str = "legacy_cutoff",
    phase1_until: int = 12_000,
    phase2_until: int = 25_000,
) -> bool:
    if not use_dense_supervision:
        return True
    if geometry_schedule == "legacy_cutoff":
        if dense_only_from_iter >= 0 and iteration > dense_only_from_iter:
            return False
        return iteration % max(int(every_n), 1) == 0
    if geometry_schedule != "persistent":
        raise ValueError(
            "geometry_schedule must be 'legacy_cutoff' or 'persistent', "
            f"got {geometry_schedule!r}"
        )
    if phase1_until < 1 or phase2_until < phase1_until:
        raise ValueError("persistent geometry phase bounds must satisfy 1 <= phase1 <= phase2")
    if iteration <= phase1_until:
        frequency = max(int(every_n), 1)
    elif iteration <= phase2_until:
        frequency = max(int(every_n) * 2, 1)
    else:
        frequency = max(int(every_n) * 4, 1)
    return iteration % frequency == 0


def geometry_prior_schedule_weight(
    iteration: int,
    *,
    geometry_schedule: str,
    phase1_until: int = 12_000,
    phase2_until: int = 25_000,
    final_weight_floor: float = 0.2,
) -> float:
    """Keep Chart geometry active through the long real-view refinement.

    The first phase is deliberately strongest, then its cadence and residual
    magnitude taper without reaching zero.  This prevents depth/normal
    supervision from disappearing after the screen bootstrap.
    """
    if not 0.0 <= final_weight_floor <= 1.0:
        raise ValueError("final geometry weight floor must lie in [0, 1]")
    if geometry_schedule == "legacy_cutoff":
        return 1.0
    if geometry_schedule != "persistent":
        raise ValueError(f"Unknown geometry schedule {geometry_schedule!r}")
    if iteration <= phase1_until:
        return 1.0
    if iteration <= phase2_until:
        return max(0.5, final_weight_floor)
    return final_weight_floor


def densification_stats_from_view(
    *,
    policy: str,
    use_dense_supervision: bool,
    is_geometry_view: bool,
) -> bool:
    """Whether a rendered view may update 2DGS topology statistics.

    A Chart view is deliberately oversampled early to provide its depth and
    plane losses.  The stock 2DGS densifier, however, also consumes that
    view's screen-space radii and position gradients.  Thus merely correcting
    the RGB loss expectation does *not* stop a small Chart set from receiving
    far more split/prune evidence than the remaining real cameras.

    ``dense_only`` keeps the geometry loss on Chart iterations but routes
    topology allocation through the uniform all-real dense camera stream.
    It is intentionally a no-op for a Chart-only run: without a dense pool,
    suppressing all topology statistics would make the control ill-defined.
    ``legacy_current`` preserves historical behavior exactly.
    """
    if policy == "legacy_current":
        return True
    if policy == "dense_only":
        return not (use_dense_supervision and is_geometry_view)
    raise ValueError(
        "densification view policy must be 'legacy_current' or 'dense_only', "
        f"got {policy!r}"
    )


def opacity_reset_due(
    iteration: int,
    *,
    opacity_reset_interval: int,
    densify_from_iter: int,
    densify_until_iter: int,
    white_background: bool,
    continue_after_densify: bool,
) -> bool:
    """Return whether an opacity reset is due under a declared schedule.

    Stock 2DGS and the released ULF/STDLoc loops place this operation inside
    the densification window.  Preserve that behavior by default.  A causal
    control can opt into continuing resets after the window in order to hold
    the reset timeline fixed while varying only topology allocation.
    """
    if iteration >= densify_until_iter and not continue_after_densify:
        return False
    return (
        iteration % opacity_reset_interval == 0
        or (white_background and iteration == densify_from_iter)
    )


def scale_chart_geometry_priors(
    losses: tuple[torch.Tensor, ...], *, weight: float
) -> tuple[torch.Tensor, ...]:
    """Apply the one explicit weight shared by Chart-exclusive priors.

    A Chart iteration contains the same RGB observation as a real dense
    camera, but it additionally contributes aligned-depth, normal, curvature,
    depth-order, and anisotropy priors.  A zero-weight ablation must remove
    only those extra residuals while preserving the camera schedule, RGB
    importance weighting, and topology updates.  Keeping this operation in a
    small helper makes that causal contract testable.
    """
    if weight < 0.0:
        raise ValueError("chart geometry prior weight must be non-negative")
    return tuple(loss * float(weight) for loss in losses)


def rgb_sampling_importance_weights(
    *,
    policy: str,
    total_iterations: int,
    dense_view_count: int,
    chart_view_count: int,
    use_dense_supervision: bool,
    geometry_view_every_n_iter: int,
    dense_only_from_iter: int,
    geometry_schedule: str = "legacy_cutoff",
    geometry_phase1_until: int = 12_000,
    geometry_phase2_until: int = 25_000,
    start_iteration: int = 0,
) -> tuple[float, float]:
    """Return RGB importance weights for chart and non-chart real cameras.

    The legacy refinement loop samples a Chart camera whenever it needs a
    chart-depth prior, then samples a camera from the full dense set on all
    other iterations.  Since the Chart cameras are also members of the dense
    set, this makes their RGB observations much more frequent than every other
    real training image.  That changes the RGB reconstruction objective even
    when every train image is technically present.

    ``all_train_importance`` retains the geometry schedule but weights RGB
    observations by ``uniform_camera_probability / actual_sampling_probability``.
    Its expectation is therefore the simple uniform all-train RGB objective
    used by the standard ULF-Loc/2DGS controls.  Geometry priors are deliberately
    not reweighted: their Chart concentration is the causal intervention.
    """
    if policy == "legacy_interleaved":
        return 1.0, 1.0
    if policy != "all_train_importance":
        raise ValueError(
            "rgb sampling policy must be 'legacy_interleaved' or "
            f"'all_train_importance', got {policy!r}"
        )
    if not use_dense_supervision:
        return 1.0, 1.0
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if dense_view_count <= 0:
        raise ValueError("dense_view_count must be positive with dense supervision")
    if chart_view_count <= 0:
        return 1.0, 1.0
    if chart_view_count > dense_view_count:
        raise ValueError("chart_view_count cannot exceed dense_view_count")

    if not 0 <= int(start_iteration) < int(total_iterations):
        raise ValueError("start_iteration must lie in [0, total_iterations)")
    chart_iterations = sum(
        geometry_iteration(
            iteration,
            use_dense_supervision=use_dense_supervision,
            every_n=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
            geometry_schedule=geometry_schedule,
            phase1_until=geometry_phase1_until,
            phase2_until=geometry_phase2_until,
        )
        for iteration in range(int(start_iteration) + 1, total_iterations + 1)
    )
    executed_iterations = int(total_iterations) - int(start_iteration)
    chart_fraction = chart_iterations / float(executed_iterations)
    dense_fraction = 1.0 - chart_fraction
    if dense_fraction <= 0.0:
        raise ValueError(
            "all_train_importance requires at least one dense-camera iteration; "
            "the current geometry schedule samples only Charts"
        )

    target_probability = 1.0 / float(dense_view_count)
    chart_probability = (
        chart_fraction / float(chart_view_count)
        + dense_fraction / float(dense_view_count)
    )
    non_chart_probability = dense_fraction / float(dense_view_count)
    return (
        float(target_probability / chart_probability),
        float(target_probability / non_chart_probability),
    )


def rgb_sampling_importance_weights_by_camera(
    *,
    policy: str,
    total_iterations: int,
    dense_camera_names: list[str],
    chart_camera_names: set[str],
    use_dense_supervision: bool,
    geometry_view_every_n_iter: int,
    dense_only_from_iter: int,
    dense_view_sampling_policy: str,
    dense_view_blocks: list[list[int]] | None = None,
    geometry_schedule: str = "legacy_cutoff",
    geometry_phase1_until: int = 12_000,
    geometry_phase2_until: int = 25_000,
    start_iteration: int = 0,
) -> dict[str, float]:
    """Return per-real-camera RGB importance weights for the actual sampler.

    Spatial block balancing changes a camera's sampling probability according
    to its block population.  A single Chart/non-Chart scalar cannot correct
    that objective, so this helper uses the exact mixture of Chart geometry
    and dense/topology sampling streams.
    """
    names = list(dense_camera_names)
    if policy == "legacy_interleaved" or not use_dense_supervision:
        return {name: 1.0 for name in names}
    if policy != "all_train_importance":
        raise ValueError(f"Unknown RGB sampling policy {policy!r}")
    if total_iterations <= 0 or not names:
        raise ValueError("all_train_importance requires positive iterations and dense cameras")
    if not chart_camera_names:
        return {name: 1.0 for name in names}
    unknown_charts = chart_camera_names - set(names)
    if unknown_charts:
        raise ValueError(f"Chart cameras are absent from dense RGB set: {sorted(unknown_charts)[:3]}")
    if not 0 <= int(start_iteration) < int(total_iterations):
        raise ValueError("start_iteration must lie in [0, total_iterations)")
    geometry_count = sum(
        geometry_iteration(
            iteration,
            use_dense_supervision=use_dense_supervision,
            every_n=geometry_view_every_n_iter,
            dense_only_from_iter=dense_only_from_iter,
            geometry_schedule=geometry_schedule,
            phase1_until=geometry_phase1_until,
            phase2_until=geometry_phase2_until,
        )
        for iteration in range(int(start_iteration) + 1, total_iterations + 1)
    )
    executed_iterations = int(total_iterations) - int(start_iteration)
    geometry_fraction = geometry_count / float(executed_iterations)
    dense_fraction = 1.0 - geometry_fraction
    if dense_fraction <= 0.0:
        raise ValueError("all_train_importance requires at least one dense RGB iteration")
    if dense_view_sampling_policy == "uniform":
        dense_probability = {name: 1.0 / len(names) for name in names}
    elif dense_view_sampling_policy == "spatial_block_balanced":
        if not dense_view_blocks:
            raise ValueError("spatial_block_balanced requires non-empty dense_view_blocks")
        dense_probability = {}
        for block in dense_view_blocks:
            if not block:
                continue
            for index in block:
                dense_probability[names[index]] = 1.0 / (len(dense_view_blocks) * len(block))
        if set(dense_probability) != set(names):
            raise ValueError("spatial block sampler must assign every dense camera exactly once")
    else:
        raise ValueError(f"Unknown dense view sampling policy {dense_view_sampling_policy!r}")

    target_probability = 1.0 / len(names)
    chart_probability = 1.0 / len(chart_camera_names)
    result: dict[str, float] = {}
    for name in names:
        actual_probability = dense_fraction * dense_probability[name]
        if name in chart_camera_names:
            actual_probability += geometry_fraction * chart_probability
        result[name] = float(target_probability / actual_probability)
    return result


def resolve_active_chart_indices(
    chart_count: int,
    *,
    alignment_gate_valid: Optional[torch.Tensor] = None,
    quality_selection_active: Optional[torch.Tensor] = None,
) -> list[int]:
    """Return Charts that remain eligible for geometric work.

    The alignment gate and post-gate quality selector intentionally preserve
    tensor/camera order so later tools can audit their provenance.  That does
    *not* mean an excluded Chart is still a valid geometry-sampling target:
    drawing it merely spends a geometry iteration on zeroed depth maps.  Keep
    the immutable camera order, but make the sampling set the intersection of
    every available hard eligibility flag.
    """
    if chart_count <= 0:
        raise ValueError("chart_count must be positive")

    active = torch.ones(chart_count, dtype=torch.bool)
    for label, flag in (
        ("alignment_gate_valid", alignment_gate_valid),
        ("quality_selection_active", quality_selection_active),
    ):
        if flag is None:
            continue
        values = torch.as_tensor(flag).detach().reshape(-1).to(device="cpu")
        if values.numel() != chart_count:
            raise RuntimeError(
                f"{label} has {values.numel()} entries, expected {chart_count} Charts"
            )
        if values.dtype == torch.bool:
            valid = values
        else:
            valid = torch.isfinite(values) & (values > 0.5)
        active &= valid.to(dtype=torch.bool)

    indices = torch.nonzero(active, as_tuple=False).flatten().tolist()
    if not indices:
        raise RuntimeError("No active Charts remain after gate/quality filtering")
    return [int(index) for index in indices]


def geometry_chart_sampling_indices(
    chart_count: int,
    *,
    active_chart_indices: Iterable[int],
    policy: str,
) -> list[int]:
    """Choose input-chart slots eligible for the geometry scheduler.

    ``legacy_all_input`` is retained only as a strict ablation.  It reproduces
    the historical behavior that sampled every aligned camera, including
    Charts whose geometry had already been zeroed.  ``active_only`` is the
    correctness path: every scheduled real Chart has passed all hard filters.
    """
    if chart_count <= 0:
        raise ValueError("chart_count must be positive")
    active = sorted({int(index) for index in active_chart_indices})
    if any(index < 0 or index >= chart_count for index in active):
        raise RuntimeError("active_chart_indices contains an out-of-range Chart")
    if not active:
        raise RuntimeError("active_chart_indices must not be empty")
    if policy == "legacy_all_input":
        return list(range(chart_count))
    if policy == "active_only":
        return active
    raise ValueError(
        "chart geometry sampling policy must be 'legacy_all_input' or "
        f"'active_only', got {policy!r}"
    )


def validate_chart_camera_order(
    chart_scene_path: str | Path,
    chart_camera_names: Iterable[str],
    *,
    chart_tensor_count: int,
) -> list[str]:
    """Verify that saved chart tensors and 2DGS cameras have identical order.

    MASt3R alignment serializes per-chart depth/point tensors in the order of
    ``cameras.json``.  The 2DGS COLMAP reader independently sorts cameras by
    filename.  Those orders happen to agree for the current automatic
    Cambridge selector, but treating that coincidence as an interface is
    unsafe: a manually ordered Chart set would silently attach each depth
    prior to a different image.  Fail before initialization rather than
    optimizing corrupted geometry.
    """
    scene_path = Path(chart_scene_path)
    cameras_path = scene_path / "cameras.json"
    if not cameras_path.is_file():
        raise FileNotFoundError(
            "Chart geometry requires the MASt3R camera-order manifest: "
            f"{cameras_path}"
        )
    try:
        payload = json.loads(cameras_path.read_text(encoding="utf-8"))
        filepaths = payload["filepaths"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read MASt3R chart camera order from {cameras_path}"
        ) from error
    if not isinstance(filepaths, list):
        raise RuntimeError(
            f"MASt3R camera manifest has non-list filepaths: {cameras_path}"
        )

    expected = [Path(str(name)).stem for name in filepaths]
    actual = [Path(str(name)).stem for name in chart_camera_names]
    if len(expected) != chart_tensor_count:
        raise RuntimeError(
            "Chart tensor count does not match MASt3R camera manifest: "
            f"tensors={chart_tensor_count}, cameras={len(expected)}"
        )
    if len(actual) != chart_tensor_count:
        raise RuntimeError(
            "2DGS chart camera count does not match aligned chart tensors: "
            f"cameras={len(actual)}, tensors={chart_tensor_count}"
        )
    if expected != actual:
        mismatch = next(
            index
            for index, (saved_name, loaded_name) in enumerate(zip(expected, actual))
            if saved_name != loaded_name
        )
        raise RuntimeError(
            "Chart tensor/camera order mismatch at index "
            f"{mismatch}: MASt3R saved {expected[mismatch]!r}, but 2DGS loaded "
            f"{actual[mismatch]!r}. Reorder the Chart scene or regenerate it with "
            "a canonical filename order."
        )
    return expected


def preliminary_uses_dense_supervision(
    *,
    dense_supervision: bool,
    dense_final_only: bool,
) -> bool:
    """Whether the initial 7k refinement may sample all real dense cameras.

    This is deliberately separate from ``geometry_iteration``: before a
    dense dataset is passed to the refinement process there is no dense camera
    pool at all.  Keeping the decision here prevents a screen-only control
    from silently re-enabling dense RGB through a later command wrapper.
    """
    return bool(dense_supervision and not dense_final_only)


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


def fused_geometry_validity_mask(source_bitmask: torch.Tensor) -> torch.Tensor:
    """Return pixels anchored by plane or aligned-Chart geometry.

    Outdoor inverse-depth fusion stores a continuous precision-derived
    confidence.  That quantity is appropriate as a loss weight, but its scale
    is deliberately much lower than the legacy binary plane-confidence map.
    Initialization must therefore use source provenance rather than applying
    the old ``confidence > 0.5`` rule to it.  Mono-only pixels remain excluded
    from Gaussian initialization because they have no verified multi-view
    geometry anchor.
    """
    source_bitmask = torch.as_tensor(source_bitmask)
    return (source_bitmask.to(torch.uint8) & 0b0011) != 0


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
