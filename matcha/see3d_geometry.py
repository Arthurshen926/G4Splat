from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DepthAlignmentDiagnostics:
    accepted: bool
    reason: str
    support_pixels: int
    fit_pixels: int
    inlier_ratio: float
    alpha: float
    beta: float
    relative_rmse: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_render_support_mask(
    alpha: torch.Tensor,
    visibility: torch.Tensor,
    render_depth: torch.Tensor,
    *,
    alpha_threshold: float = 0.9,
    erosion_radius: int = 1,
) -> torch.Tensor:
    """Return pixels supported by real views and a valid baseline render."""
    alpha = alpha.squeeze()
    visibility = visibility.squeeze().to(device=alpha.device)
    render_depth = render_depth.squeeze().to(device=alpha.device)
    if alpha.shape != visibility.shape or alpha.shape != render_depth.shape:
        raise ValueError(
            "alpha, visibility, and render_depth must have the same shape; "
            f"got {tuple(alpha.shape)}, {tuple(visibility.shape)}, "
            f"and {tuple(render_depth.shape)}"
        )

    support = (
        (alpha >= alpha_threshold)
        & (visibility > 0.5)
        & torch.isfinite(render_depth)
        & (render_depth > 0)
    )
    if erosion_radius > 0:
        kernel = 2 * int(erosion_radius) + 1
        unsupported = (~support).float()[None, None]
        support = F.max_pool2d(
            unsupported,
            kernel_size=kernel,
            stride=1,
            padding=erosion_radius,
        )[0, 0] < 0.5
    return support


def _fit_affine(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_mean = x.mean()
    y_mean = y.mean()
    variance = (x - x_mean).square().mean()
    if not torch.isfinite(variance) or variance <= 1e-12:
        raise RuntimeError("degenerate disparity variance")
    beta = ((x - x_mean) * (y - y_mean)).mean() / variance
    alpha = y_mean - beta * x_mean
    return alpha, beta


def robust_align_inverse_depth(
    disparity: torch.Tensor,
    render_depth: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    min_samples: int = 512,
    max_samples: int = 200_000,
    max_relative_rmse: float = 0.2,
    max_iterations: int = 4,
    outlier_sigma: float = 3.0,
) -> tuple[torch.Tensor, DepthAlignmentDiagnostics]:
    """Robustly fit ``1 / depth = alpha + beta * disparity`` on supported pixels."""
    disparity = disparity.squeeze().float()
    render_depth = render_depth.squeeze().to(device=disparity.device, dtype=torch.float32)
    support_mask = support_mask.squeeze().to(device=disparity.device, dtype=torch.bool)
    if disparity.shape != render_depth.shape or disparity.shape != support_mask.shape:
        raise ValueError("disparity, render_depth, and support_mask must have matching shapes")

    valid = (
        support_mask
        & torch.isfinite(disparity)
        & torch.isfinite(render_depth)
        & (disparity > 0)
        & (render_depth > 0)
    )
    support_pixels = int(valid.sum().item())

    def rejected(reason: str) -> tuple[torch.Tensor, DepthAlignmentDiagnostics]:
        diagnostics = DepthAlignmentDiagnostics(
            accepted=False,
            reason=reason,
            support_pixels=support_pixels,
            fit_pixels=0,
            inlier_ratio=0.0,
            alpha=float("nan"),
            beta=float("nan"),
            relative_rmse=float("inf"),
        )
        return torch.zeros_like(disparity), diagnostics

    if support_pixels < min_samples:
        return rejected(f"insufficient support ({support_pixels} < {min_samples})")

    x = disparity[valid]
    y = render_depth[valid].reciprocal()
    if x.numel() > max_samples:
        indices = torch.linspace(
            0,
            x.numel() - 1,
            steps=max_samples,
            device=x.device,
        ).round().long()
        x = x[indices]
        y = y[indices]

    try:
        alpha, beta = _fit_affine(x, y)
    except RuntimeError as error:
        return rejected(str(error))

    inliers = torch.ones_like(x, dtype=torch.bool)
    for _ in range(max(int(max_iterations), 1)):
        residual = y - (alpha + beta * x)
        residual_center = residual.median()
        mad = (residual - residual_center).abs().median()
        threshold = torch.clamp(1.4826 * outlier_sigma * mad, min=1e-5)
        next_inliers = (residual - residual_center).abs() <= threshold
        if int(next_inliers.sum().item()) < min_samples:
            break
        inliers = next_inliers
        try:
            alpha, beta = _fit_affine(x[inliers], y[inliers])
        except RuntimeError:
            break

    fit_pixels = int(inliers.sum().item())
    residual = y[inliers] - (alpha + beta * x[inliers])
    rmse = residual.square().mean().sqrt()
    relative_rmse = rmse / y[inliers].abs().median().clamp_min(1e-6)
    inlier_ratio = fit_pixels / max(int(x.numel()), 1)

    reason = "ok"
    accepted = True
    if not torch.isfinite(alpha) or not torch.isfinite(beta):
        accepted = False
        reason = "non-finite affine parameters"
    elif beta <= 0:
        accepted = False
        reason = "non-positive disparity scale"
    elif not torch.isfinite(relative_rmse) or relative_rmse > max_relative_rmse:
        accepted = False
        reason = (
            f"relative inverse-depth RMSE {float(relative_rmse):.4f} "
            f"> {max_relative_rmse:.4f}"
        )

    denominator = alpha + beta * disparity
    positive = torch.isfinite(denominator) & (denominator > 1e-6)
    aligned_depth = torch.zeros_like(disparity)
    if accepted:
        aligned_depth[positive] = denominator[positive].reciprocal()
        supported_depths = render_depth[valid]
        lower = torch.quantile(supported_depths, 0.01).clamp_min(1e-4) / 4.0
        upper = torch.quantile(supported_depths, 0.99).clamp_min(lower * 2.0) * 4.0
        aligned_depth = aligned_depth.clamp(min=float(lower), max=float(upper))

    diagnostics = DepthAlignmentDiagnostics(
        accepted=accepted,
        reason=reason,
        support_pixels=support_pixels,
        fit_pixels=fit_pixels,
        inlier_ratio=float(inlier_ratio),
        alpha=float(alpha),
        beta=float(beta),
        relative_rmse=float(relative_rmse),
    )
    return aligned_depth, diagnostics
