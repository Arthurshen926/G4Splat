"""Preserve native depth samples across area-reduction cells.

Layers are proposals, not surfaces or independently verified geometry.  More
than two separated modes are marked unresolved rather than averaged together.
"""
from __future__ import annotations

import numpy as np


def reduce_depth_layers(depth, valid, shape, *, separation_ratio=1.10):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    h, w = map(int, shape)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError('Depth and validity must be aligned HxW arrays')
    if h <= 0 or w <= 0 or h > depth.shape[0] or w > depth.shape[1]:
        raise ValueError('Layer reduction requires positive non-upsampling dimensions')
    if not np.isfinite(separation_ratio) or separation_ratio <= 1:
        raise ValueError('Layer separation ratio must exceed one')
    native_h,native_w=depth.shape
    ylo=np.arange(h)*native_h//h; yhi=((np.arange(h)+1)*native_h+h-1)//h
    xlo=np.arange(w)*native_w//w; xhi=((np.arange(w)+1)*native_w+w-1)//w
    dy,dx=int((yhi-ylo).max()),int((xhi-xlo).max())
    valid = valid & np.isfinite(depth) & (depth > 0)
    clean=np.where(valid,depth,np.inf)
    if native_h%h==0 and native_w%w==0:
        values=clean.reshape(h,dy,w,dx).transpose(0,2,1,3).reshape(h,w,-1)
    else:
        # Same floor/ceil bins as adaptive average pooling (PyTorch area).
        # Neighboring bins may overlap; original native pixel identity remains
        # attached so a downstream fusion can deduplicate those observations.
        yy=ylo[:,None]+np.arange(dy)[None,:]
        xx=xlo[:,None]+np.arange(dx)[None,:]
        inside=(yy[:,None,:,None]<yhi[:,None,None,None])&(xx[None,:,None,:]<xhi[None,:,None,None])
        values=np.where(inside,clean[np.minimum(yy,native_h-1)[:,None,:,None],
                                      np.minimum(xx,native_w-1)[None,:,None,:]],np.inf).reshape(h,w,-1)
    cell_area=(yhi-ylo)[:,None]*(xhi-xlo)[None,:]
    order = np.argsort(values, axis=-1, kind='stable')
    ordered = np.take_along_axis(values, order, axis=-1)
    finite = np.isfinite(ordered)
    count = finite.sum(-1)
    safe = np.where(finite, ordered, 1.)
    gaps = (np.log(safe[..., 1:]) - np.log(safe[..., :-1])) > np.log(separation_ratio)
    gaps &= finite[..., 1:]
    modes = gaps.sum(-1) + (count > 0)
    split = np.argmax(gaps, axis=-1) + 1 if values.shape[-1] > 1 else np.ones((h, w), int)
    rank = np.arange(values.shape[-1])
    front = finite & ((modes <= 1)[..., None] | (rank < split[..., None]))
    back = finite & (modes == 2)[..., None] & (rank >= split[..., None])
    # Even a smooth but large within-cell depth ramp cannot become a single
    # measured surface. Preserve samples but withhold continuous-base authority.
    low = np.where(count > 0, ordered[..., 0], 1.)
    high = np.take_along_axis(safe, np.maximum(count - 1, 0)[..., None], -1)[..., 0]
    mixed = (count > 0) & (high / low > separation_ratio)
    result = dict(valid_fraction=(count / cell_area).astype(np.float32),
                  mixed=mixed, unresolved=modes > 2, mode_count=modes.astype(np.uint8),
                  continuous_valid=(count >= .5 * cell_area) & ~mixed)
    yy, xx = np.indices((h, w))
    for name, member in [('front', front), ('back', back)]:
        n = member.sum(-1)
        # Pick an actual native sample; never put the representative on the
        # cell-centre ray unless that was its original ray.
        start = np.where(name == 'front', 0, split)
        pick = np.minimum(start + np.maximum(n - 1, 0) // 2, values.shape[-1] - 1)
        offset = np.take_along_axis(order, pick[..., None], -1)[..., 0]
        z = np.take_along_axis(ordered, pick[..., None], -1)[..., 0]
        accepted = (n > 0) & (modes <= 2)
        result[name + '_depth'] = np.where(accepted, z, 0.).astype(np.float32)
        result[name + '_fraction'] = np.where(accepted, n / cell_area, 0.).astype(np.float32)
        result[name + '_native_x'] = np.where(accepted, xlo[xx] + offset % dx, -1).astype(np.int32)
        result[name + '_native_y'] = np.where(accepted, ylo[yy] + offset // dx, -1).astype(np.int32)
    return result
