"""Diagnostic depth-source calibration against a frozen visible rigid scene."""
import math
import torch
import torch.nn.functional as F


def calibrated_canopy_evidence(evidence, scale):
    """Convert depth before constructing metric-capped hit intervals.

    Preserve the global rigid/Chart authority. The calibrated depth is a
    separate canopy field; multiplying already capped intervals would scale
    the physical thickness cap and incorrectly turn scale into material.
    """
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Canopy depth calibration must be finite and positive')
    if 'moge3_canopy_depth_m' in evidence:
        raise ValueError('Canopy evidence must not be calibrated twice')
    return {**evidence, 'moge3_canopy_depth_m': evidence['moge3_depth_m'] * scale}


@torch.no_grad()
def fit_rigid_depth_scale(moge_depth,rigid_depth,rigid_alpha,known_rigid,minimum_pixels=512):
    shape=known_rigid.shape
    depth=moge_depth.reshape(shape);wall=rigid_depth.reshape(shape);alpha=rigid_alpha.reshape(shape)
    core=(1-F.max_pool2d((~known_rigid).float()[None,None],5,1,2)[0,0])>.5
    keep=core&torch.isfinite(depth)&torch.isfinite(wall)&(depth>0)&(wall>0)&(alpha>=.98)
    count=int(keep.sum())
    if count<minimum_pixels:return {'scale':1.,'applied':False,'pixels':count,'reason':'insufficient_visible_rigid'}
    ratio=(wall[keep]/depth[keep]).log()
    center=ratio.median();scatter=1.4826*(ratio-center).abs().median()
    scale=float(center.exp())
    # A consistent anchor ratio >2 is evidence, not an error merely because
    # it differs from the initial global guess. Reject inconsistent anchors,
    # not the magnitude of a data-supported positive metric conversion.
    accepted=math.isfinite(scale) and scale>0 and float(scatter)<=.20
    return {'scale':scale if accepted else 1.,'applied':accepted,'pixels':count,
            'fitted_scale':scale,'robust_log_scatter':float(scatter),
            'reason':'frozen_rigid_single_scale_diagnostic' if accepted else 'inconsistent_or_out_of_range'}
