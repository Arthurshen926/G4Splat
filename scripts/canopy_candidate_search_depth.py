"""A blocking surface bounds a search opportunity, not a true leaf depth."""
import torch


def blocking_aware_search_depth(prior, surface_depth, surface_alpha):
    if prior.shape != surface_depth.shape or prior.shape != surface_alpha.shape:
        raise ValueError('Aligned source-ray hypotheses and native blocker required')
    prior, surface_depth, surface_alpha = [x.detach() for x in (prior, surface_depth, surface_alpha)]
    known = (torch.isfinite(surface_depth) & (surface_depth > .21)
             & torch.isfinite(surface_alpha) & (surface_alpha >= .95)
             & torch.isfinite(prior) & (prior > .2))
    near = .98*surface_depth
    changed = known & (near < prior)
    return torch.where(changed, near, prior), changed
