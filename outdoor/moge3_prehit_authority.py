"""Separate a monocular canopy location prior from observed empty-space evidence."""
import torch


def observed_rigid_prehit_authority(task, rigid_depth, rigid_alpha):
    rigid=task['p_rigid'].detach();canopy=task['p_canopy'].detach()
    depth=rigid_depth.detach().reshape_as(rigid)
    alpha=rigid_alpha.detach().reshape_as(rigid)
    known=(torch.isfinite(depth)&(depth>0)&torch.isfinite(alpha)&(alpha>=.95)
           &torch.isfinite(rigid)&(rigid>=.5)&torch.isfinite(canopy)&(canopy<=1e-4))
    return torch.where(known,rigid.clamp(0,1),torch.zeros_like(rigid))
