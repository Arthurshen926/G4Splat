"""Training-only relative RGB floor, calibrated to the initial absolute gradient."""
import torch

EPSILON = 1./255.


def _inputs(surface, target, canopy):
    if (surface.shape != target.shape or surface.shape != (3, *canopy.shape)
            or canopy.dtype != torch.bool or not torch.isfinite(surface).all()
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Aligned finite nonnegative observed RGB and canopy required')
    return surface.clamp_min(0), target.detach()


@torch.no_grad()
def initial_gradient_normalization(surface, target, canopy):
    surface, target = _inputs(surface, target, canopy)
    if not canopy.any(): return 1.
    absolute = (surface-target).relu()
    relative = ((surface+EPSILON).log()-(target+EPSILON).log()).relu()
    absolute_gradient = (2*absolute[:, canopy]).sum()
    relative_gradient = (2*relative/(surface+EPSILON))[:, canopy].sum()
    if relative_gradient > 0:
        scale = absolute_gradient/relative_gradient
    else:
        # Initially feasible views must still penalize future violations.
        # Local squared-log/absolute equivalence near the observed radiance.
        scale = (target[:, canopy]+EPSILON).square().mean()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError('Finite positive initial radiance-gradient normalization required')
    return float(scale)


def relative_surface_rgb_feasibility_loss(surface, target, canopy, initial_scale):
    surface, target = _inputs(surface, target, canopy)
    if not 0 < initial_scale < float('inf'):
        raise ValueError('Frozen training-source normalization required')
    excess = ((surface+EPSILON).log()-(target+EPSILON).log()).relu()
    return initial_scale*(excess.square().mean(0)*canopy).sum()/canopy.sum().clamp_min(1)
