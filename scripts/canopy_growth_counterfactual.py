"""Finite shrink probes from training-only derivatives; never exported repairs."""
from contextlib import contextmanager
import torch


def eligible_shrink(data, code, snapshot_sha256, training_views, excluded_views):
    report = data['report']
    if (report['snapshot_sha256'] != snapshot_sha256
            or report['training_views'] != training_views
            or report['excluded_views'] != excluded_views
            or set(training_views) & set(excluded_views)
            or not torch.equal(data['scale_code'].to(code), code.detach())):
        raise ValueError('Growth evidence must match exact snapshot, code and training scope')
    tree, risk = (data[k].to(code.device) for k in ('tree_gradient', 'risk_gradient'))
    tc, rc = (data[k].to(code.device) for k in ('tree_nonzero_views', 'risk_nonzero_views'))
    for tensor in (tree, risk, tc, rc):
        if tensor.shape != code.shape or not torch.isfinite(tensor).all():
            raise ValueError('Finite aligned growth evidence required')
    if (tc < 0).any() or (rc < 0).any() or (tc > len(training_views)).any() or (rc > len(training_views)).any():
        raise ValueError('Invalid distinct-view counts')
    return (code.detach() > 0) & (tree > 0) & (risk > 0) & (tc >= 2) & (rc >= 2)


@contextmanager
def shrink_probe(code, eligible, fraction):
    if (fraction not in (0., .25, 1.) or eligible.dtype != torch.bool
            or eligible.shape != code.shape):
        raise ValueError('Explicit finite probe and aligned selection required')
    before = code.detach().clone()
    try:
        with torch.no_grad():
            code[eligible] = before[eligible] * (1 - fraction)
        yield 1.
    finally:
        with torch.no_grad():
            code.copy_(before)
        if not torch.equal(code.detach(), before):
            raise RuntimeError('Growth probe failed to restore scale codes')
