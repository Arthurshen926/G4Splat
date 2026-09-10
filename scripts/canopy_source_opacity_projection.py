"""Opt-in source opacity projection; frozen opacity must never be clamped."""
import math
import torch


@torch.no_grad()
def project_source_opacity_(logits, eligible, *, enabled):
    if not enabled:
        return
    logits[eligible] = logits[eligible].clamp(math.log(1e-6/(1-1e-6)), math.log(.995/.005))
