"""Box-constrained majorized least squares for fixed-geometry leaf colors."""
import torch


@torch.no_grad()
def majorized_color_update(current,numerator,denominator,eligible,upper):
    """D = W^T(mu * row_sum(W)) majorizes W^T diag(mu) W.

Using a larger row sum (including frozen primitives) is conservative. All
observations must be accumulated at the SAME colors before this Jacobi update.
"""
    if current.shape!=numerator.shape or current.shape!=upper.shape or current.shape[-1]!=3:
        raise ValueError('Aligned RGB arrays required')
    if denominator.shape!=(len(current),) or eligible.shape!=denominator.shape:
        raise ValueError('Aligned per-leaf denominator and eligibility required')
    active=eligible & (denominator>1.e-12) & torch.isfinite(denominator) & torch.isfinite(numerator).all(dim=1)
    result=current.clone()
    result[active]=(current[active]+numerator[active]/denominator[active,None]).clamp_min(0)
    result[active]=torch.minimum(result[active],upper[active])
    return result,active
