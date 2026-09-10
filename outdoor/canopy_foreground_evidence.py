"""Conservative appearance evidence for diagnostic leaf proposal selection.

An object mask includes real gaps. A strong mismatch with a frozen background
is positive occluder evidence, not a depth measurement or an opacity target.
Weak mismatch stays UNKNOWN, never known free space. Not a render-time gate.
"""
import torch
import torch.nn.functional as F


def foreground_darkening_bound(observed,background,slack=.02):
    """Necessary alpha bound for an opaque background and nonnegative front RGB.

    Slack represents background/photometric model error, not material opacity.
    Neither this bound nor its RGB inputs should receive optimization gradients.
    """
    if observed.shape!=background.shape or observed.shape[0]!=3:
        raise ValueError('Aligned RGB required')
    b=background.detach();c=observed.detach()
    ratio=torch.where(b>.05,(b-c-slack)/b.clamp_min(.05),torch.zeros_like(b))
    return ratio.amax(dim=0).clamp(0,.99)


def mixed_order_darkening_bound(observed, background, radiance_ceiling, slack=.02):
    """Necessary visible-leaf alpha even when leaves interleave fixed surfaces.

    Inserting nonnegative leaves removes exactly their visible alpha from the
    fixed-background (surfaces plus sky) weight budget. Removed RGB is at most
    that alpha times the largest background primitive/sky RGB in each channel.
    Thus B-C <= A_leaf*M, irrespective of insertion depth. Unlike the layered
    B denominator, M must bound EVERY background contributor, not merely the
    already-composited pixel. Use raw, unclipped renderer RGB and detached M.
    This is weaker than the front-only bound, not an opacity measurement.
    A lower surrogate formed by capping each fixed contributor before blending
    is also valid with its cap as M; clamping the final blended image is NOT.
    """
    if observed.shape != background.shape or observed.ndim != 3 or observed.shape[0] != 3:
        raise ValueError('Aligned raw RGB required')
    ceiling=torch.as_tensor(radiance_ceiling,device=background.device,dtype=background.dtype).detach()
    if ceiling.shape == (3,):ceiling=ceiling[:,None,None]
    if ceiling.shape not in ((3,1,1),background.shape):
        raise ValueError('Channel or pixelwise contributor radiance ceiling required')
    slack=torch.as_tensor(slack,device=background.device,dtype=background.dtype).detach()
    if (not torch.isfinite(ceiling).all() or (ceiling<0).any() or not torch.isfinite(slack).all()
            or (slack<0).any() or not torch.isfinite(background).all() or not torch.isfinite(observed).all()):
        raise ValueError('Finite nonnegative radiance ceiling and model-error slack required')
    if (background.detach()>ceiling+1.e-5).any():
        raise ValueError('Ceiling does not even bound the composed background')
    ratio=(background.detach()-observed.detach()-slack)/ceiling.clamp_min(1.e-6)
    return ratio.amax(dim=0).clamp(0,.99)


def foreground_darkening_loss(front_alpha,required_alpha,applicable):
    """Grow only a demonstrated extinction deficit; real gaps get no target."""
    if front_alpha.shape!=required_alpha.shape or applicable.shape!=front_alpha.shape:
        raise ValueError('Aligned scalar alpha and permission maps required')
    target_tau=-torch.log1p(-required_alpha.detach().clamp(0,.99))
    tau=-torch.log1p(-front_alpha.clamp(0,1-1.e-5))
    deficit=(torch.log(target_tau+1.e-6)-torch.log(tau+1.e-6)).clamp_min(0)
    mask=applicable.detach()&(required_alpha.detach()>0)
    values=F.smooth_l1_loss(deficit,torch.zeros_like(deficit),reduction='none')
    return (values*mask).sum()/mask.sum().clamp_min(1)


@torch.no_grad()
def foreground_contrast_evidence(rgb,background,canopy,rigid,rigid_alpha,profile=None,minimum_pixels=512):
    residual=(rgb-background).abs().mean(dim=0)
    if profile is None:
        core=(1-F.max_pool2d((~rigid).float()[None,None],5,1,2)[0,0])>.5
        anchors=core&(rigid_alpha.reshape_as(rigid)>=.98)&torch.isfinite(residual)
        count=int(anchors.sum())
        # Estimate the background model's own error before claiming an occluder.
        profile={'applied':count>=minimum_pixels,'anchor_pixels':count,
                 'threshold':float(torch.quantile(residual[anchors],.90))+.02 if count>=minimum_pixels else None,
                 'rule':'rigid_residual_q90_plus_0.02__weak_canopy_is_unknown'}
    positive=canopy.clone()
    if profile['applied']:
        positive &= torch.isfinite(residual)&(residual>profile['threshold'])
    return positive,profile
