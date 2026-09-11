"""Color-only lower error bound at FIXED geometry/opacity, not a target."""
import torch


def nonnegative_color_floor(black, current, target, mask, *, metric_domain='raw'):
    """Allow arbitrary nonnegative leaf radiance, not just unit RGB."""
    if not (black.shape == current.shape == target.shape == (3, *mask.shape)):
        raise ValueError('Aligned CHW images required')
    if not all(torch.isfinite(t).all() for t in (black, current, target)) or (black > current+2e-5).any():
        raise ValueError('Removing nonnegative leaf radiance must not brighten pixels')
    if metric_domain not in ('raw','clamped_unit_rgb'):raise ValueError('Explicit supported metric domain required')
    if metric_domain=='clamped_unit_rgb':black,current=black.clamp(0,1),current.clamp(0,1)
    if not mask.any(): return dict(pixels=0)
    error = (current-target).square().mean(0)[mask]
    lower = (black-target).clamp_min(0).square().mean(0)[mask]
    return dict(metric_domain=metric_domain,pixels=int(mask.sum()), below_black_floor_fraction=float(((target < black-1/255).any(0))[mask].float().mean()),
                current_mse=float(error.mean()), minimum_nonnegative_color_mse=float(lower.mean()),
                irreducible_error_fraction=float(lower.sum()/error.sum().clamp_min(1e-12)),
                scope='all_persistent_leaf_colors_relaxed__fixed_opacity_geometry__not_training_truth')


def color_feasibility(black, white, current, target, mask):
    if not (black.shape == white.shape == current.shape == target.shape == (3, *mask.shape)):
        raise ValueError('Aligned CHW images required')
    width = white-black
    if not all(torch.isfinite(t).all() for t in (black, white, current, target)):
        raise ValueError('Finite native renders required')
    if width.min() < -2e-5 or (width-width.mean(0, keepdim=True)).abs().max() > 2e-5:
        raise ValueError('Color-only probe must have scalar, nonnegative native transmittance')
    if ((current < black-2e-5) | (current > white+2e-5)).any():
        raise ValueError('Current candidate colors do not obey unit-RGB premise')
    if not mask.any(): return dict(pixels=0)
    # An independently chosen color per channel and pixel is MORE permissive
    # than any shared Gaussian colors. Its residual is a necessary lower bound,
    # not a realizable scene, achievable quality claim, or opacity supervisor.
    closest = torch.maximum(black, torch.minimum(white, target))
    minimum_error = (closest-target).square().mean(0)[mask]
    current_error = (current-target).square().mean(0)[mask]
    unreachable = ((target < black-1/255) | (target > white+1/255)).any(0)
    return dict(pixels=int(mask.sum()), unreachable_color_fraction=float(unreachable[mask].float().mean()),
                mean_candidate_transmittance_weight=float(width.mean(0)[mask].mean()),
                current_mse=float(current_error.mean()), minimum_color_only_mse=float(minimum_error.mean()),
                irreducible_error_fraction=float(minimum_error.sum()/current_error.sum().clamp_min(1e-12)),
                scope='per_pixel_color_relaxation__fixed_geometry_and_opacity__not_training_truth')
