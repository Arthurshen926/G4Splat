"""Experimental observed-image risk budget plus conservative wall visibility.

The immutable source defines an error budget, not an RGB target. Improvements
below that budget get ZERO reward here. This is still a conditional soft loss,
not proof of geometry or a guarantee that held-out building views cannot worsen.
"""
import torch


def observed_rgb_risk(rgb,target,reference,mask):
    if rgb.shape!=target.shape or rgb.shape!=reference.shape or rgb.shape!=(3,*mask.shape):
        raise ValueError('Aligned observed RGB and immutable error budget required')
    if mask.dtype!=torch.bool:raise ValueError('Explicit observation mask required')
    before=(reference.detach()-target.detach()).abs().mean(0)
    after=(rgb-target.detach()).abs().mean(0)
    # relu has zero derivative at equality; clamp_min would reward some
    # improvement directions right on the zero-risk boundary.
    return (torch.relu(after-before)*mask).sum()/mask.sum().clamp_min(1)


def conservative_visible_rigid_mask(reference,target,reference_alpha,regions):
    alpha=reference_alpha.detach().reshape_as(regions['rigid'])
    close=(reference.detach()-target.detach()).abs().mean(0)<=.05
    return regions['rigid']&~regions['hard']&(alpha>=.95)&close


def visible_rigid_loss(alpha,reference_alpha,reference,target,regions):
    mask=conservative_visible_rigid_mask(reference,target,reference_alpha,regions)
    delta=reference_alpha.detach().reshape_as(mask)-alpha.reshape_as(mask)
    return (torch.relu(delta).square()*mask).sum()/mask.sum().clamp_min(1)
