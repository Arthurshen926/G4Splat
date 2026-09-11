"""Invalidate stale fixed-ray targets; this does not establish visibility."""
import torch


@torch.no_grad()
def fixed_ray_validity(xyz, world_view, anchor_pixels, focal, center, shape, max_shift=.5):
    """Only retain targets whose original pixel footprint still contains the center.

    No gradients through discrete assignment. Out-of-frame, behind-camera and
    nonfinite projections are invalid. A valid result is NOT proof of occlusion
    ordering, same-leaf correspondence, or joint multiview support.
    """
    if max_shift<0:raise ValueError('Nonnegative pixel tolerance required')
    camera=xyz@world_view[:3,:3]+world_view[3,:3]
    uv=camera[:,:2]/camera[:,2,None].clamp_min(1e-8)
    pixel=uv*torch.as_tensor(focal,device=xyz.device)+torch.as_tensor(center,device=xyz.device)-.5
    h,w=shape
    return (torch.isfinite(camera).all(-1)&(camera[:,2]>0)&
            (pixel[:,0]>=-.5)&(pixel[:,0]<w-.5)&
            (pixel[:,1]>=-.5)&(pixel[:,1]<h-.5)&
            ((pixel-anchor_pixels).abs().amax(-1)<=max_shift))
