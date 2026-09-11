"""Necessary extinction bound for measured rigid light BEFORE a depth."""
import numpy as np


def optical_prefix_lower_bound(prefix_rgb, target_rgb, total_rigid_rgb, *, tolerance=.03, leakage_fraction=1e-4):
    prefix,target,total=[np.asarray(x,dtype=np.float64) for x in (prefix_rgb,target_rgb,total_rigid_rgb)]
    if (any(x.shape!=(3,) or not np.isfinite(x).all() or (x<0).any() for x in (prefix,target,total))
            or not np.isfinite(tolerance) or tolerance<=0
            or not np.isfinite(leakage_fraction) or not 0<=leakage_fraction<=.01):
        raise ValueError('Finite nonnegative RGB and explicit bounded tolerance required')
    conservative=np.maximum(prefix-leakage_fraction*total,0.)
    return max(float(np.log(np.maximum(conservative,1e-12)/(target+tolerance)).max()),0.)
