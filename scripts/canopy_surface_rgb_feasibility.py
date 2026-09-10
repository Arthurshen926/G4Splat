"""Optional RGB-derived necessary condition, never an opacity target.

At fixed background appearance, nonnegative foliage radiance cannot cancel a
surface contribution already brighter than the observation. This auxiliary
is useful only as an explicitly authorized controlled optimization experiment.
"""
import torch


class BlackRadianceView:
    """Wrap a render view without broadening its parameter/evidence authority."""
    def __init__(self, wrapped): self.wrapped = wrapped
    def __len__(self): return len(self.wrapped)
    def __getattr__(self, name): return getattr(self.wrapped, name)
    def conditioned_state(self, temporal_code, *, include_dynamic, **kwargs):
        xyz, features, opacity = self.wrapped.conditioned_state(
            temporal_code, include_dynamic=include_dynamic, **kwargs)
        black = torch.zeros_like(features)
        black[:, 0] = -.5/.28209479177387814
        return xyz, black, opacity


def surface_rgb_feasibility_loss(surface_rgb, target, canopy):
    if surface_rgb.shape != target.shape or surface_rgb.shape != (3, *canopy.shape) or canopy.dtype != torch.bool:
        raise ValueError('Aligned RGB and explicit canopy mask required')
    excess = (surface_rgb-target.detach()).clamp_min(0)
    return (excess.square().mean(0)*canopy).sum()/canopy.sum().clamp_min(1)
