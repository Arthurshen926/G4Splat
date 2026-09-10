"""Contribution-weighted peak opacity, not semantic/geometry verification."""
import torch


def weighted_opacity_summary(opacity, weight):
    opacity, weight = opacity.reshape(-1), weight.reshape(-1)
    if opacity.shape != weight.shape or (weight < 0).any() or not torch.isfinite(weight).all():
        raise ValueError('Aligned nonnegative contribution required')
    total = weight.sum(); denominator = total.clamp_min(1e-12)
    return dict(contribution=float(total), contributing_rows=int((weight > 1e-6).sum()),
                weighted_peak_opacity=float((weight*opacity).sum()/denominator) if total > 0 else None,
                contribution_from_peak_above_09=float((weight*(opacity > .9)).sum()/denominator),
                contribution_from_peak_above_099=float((weight*(opacity > .99)).sum()/denominator),
                contribution_from_peak_below_01=float((weight*(opacity < .1)).sum()/denominator),
                warning='Peak opacity is not pixel alpha; footprint tails and preceding occlusion still matter')


def weighted_size_summary(ratio, weight):
    ratio, weight = ratio.reshape(-1), weight.reshape(-1)
    if ratio.shape != weight.shape or not torch.isfinite(ratio).all() or not torch.isfinite(weight).all() or (ratio <= 0).any() or (weight < 0).any():
        raise ValueError('Finite positive size ratios and aligned nonnegative contributions required')
    total = weight.sum(); denominator = total.clamp_min(1e-12)
    return dict(weighted_size_ratio=float((ratio*weight).sum()/denominator) if total > 0 else None,
                contribution_from_size_above_18=float((weight*(ratio > 1.8)).sum()/denominator),
                contribution_from_size_below_08=float((weight*(ratio < .8)).sum()/denominator),
                scope='actual_ROI_contribution_weighted_isotropic_size__not_geometry_truth')
