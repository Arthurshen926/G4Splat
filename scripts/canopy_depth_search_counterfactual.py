"""Ephemeral depth-search range probe; never a trained model or depth truth."""
import math
import torch


@torch.no_grad()
def reseed_stratified_depth_(candidate, views, seed_audit, *, old_low, new_low, high):
    if not (.1 <= new_low < high <= 1.5 and .1 <= old_low < high):
        raise ValueError('Bounded search ratios required')
    offset = 0
    for row in seed_audit:
        rays = row['rays']; count = rays*3
        if not count: continue
        u = torch.rand((rays, 3), generator=torch.Generator().manual_seed(37001+row['index']),
                       dtype=candidate.xyz.dtype)
        unit = (torch.arange(3, dtype=candidate.xyz.dtype)[None]+u)/3
        old_ratio = (math.log(old_low)+(math.log(high)-math.log(old_low))*unit).exp()
        new_ratio = (math.log(new_low)+(math.log(high)-math.log(new_low))*unit).exp()
        factor = (new_ratio/old_ratio).reshape(-1, 1).to(candidate.xyz.device)
        center = views[row['index']].camera_center
        candidate.xyz[offset:offset+count] = center+(candidate.xyz[offset:offset+count]-center)*factor
        candidate.scales[offset:offset+count] *= factor
        offset += count
    if offset != len(candidate.xyz): raise ValueError('Candidate order/capacity does not match seed audit')
    return dict(scope='ephemeral_geometry_search_counterfactual__not_trained_or_verified',
                old_min_ratio=old_low, new_min_ratio=new_low, max_ratio=high,
                preserved='source ray projection, angular footprint, colors, opacity; source scene unchanged')
