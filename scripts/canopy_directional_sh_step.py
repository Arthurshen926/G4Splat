"""Actual directional-SH step scaling, not Adam-cancelled gradient scaling."""
import torch


@torch.no_grad()
def rescale_directional_sh_step_(features, previous_rest, eligible, multiplier):
    if not 0 <= multiplier <= 1 or eligible.dtype != torch.bool or eligible.shape != features.shape[:1]:
        raise ValueError('Bounded multiplier and explicit source authority required')
    if previous_rest.shape != features[eligible, 1:].shape:
        raise ValueError('Aligned pre-Adam directional coefficients required')
    if multiplier == 1: return
    if multiplier == 0:
        features[eligible, 1:] = previous_rest
    else:
        features[eligible, 1:] = previous_rest+multiplier*(features[eligible, 1:]-previous_rest)
