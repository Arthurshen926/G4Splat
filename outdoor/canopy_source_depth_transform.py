"""Diagnostic coherent depth correction; no camera-dependent rendering state."""
import torch


def source_depth_transform(xyz, log_scales, origins, group_index, log_factors, owner):
    """Move along source bearings and preserve source-camera angular footprint.

    Scaling center distance and all covariance axes together preserves the
    projected conic in the source camera. Opacity, colors and rigid geometry
    are not inputs. A bound on log_factors belongs to the optimizer policy.
    """
    factor = log_factors[group_index]
    relative = xyz - origins[group_index]
    proposed = origins[group_index] + relative * factor.exp()[:, None]
    return (torch.where(owner[:, None], proposed, xyz),
            torch.where(owner[:, None], log_scales + factor[:, None], log_scales))


def source_depth_chain_gradient(xyz_gradient, scale_gradient, xyz, origins,
                                group_index, group_count, owner):
    row_gradient = ((xyz_gradient * (xyz - origins[group_index])).sum(dim=1)
                    + scale_gradient.sum(dim=1)) * owner
    result = row_gradient.new_zeros(group_count)
    return result.scatter_add(0, group_index, row_gradient)
