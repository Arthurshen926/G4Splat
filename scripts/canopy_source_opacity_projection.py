"""Opt-in source opacity projection; frozen opacity must never be clamped."""
import math
import torch

# Stay above the native per-pixel1e-6 cutoff so a projected live leaf still
# receives opacity gradients around its centre after an unfavorable update.
LIVE_OPACITY_FLOOR = 1e-5


@torch.no_grad()
def project_source_opacity_(logits, eligible, *, enabled):
    if not enabled:
        return
    logits[eligible] = logits[eligible].clamp(math.log(LIVE_OPACITY_FLOOR/(1-LIVE_OPACITY_FLOOR)), math.log(.995/.005))
