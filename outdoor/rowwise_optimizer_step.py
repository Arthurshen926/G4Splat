"""Row-local Adam learning-rate corrections without gradient rescaling."""
import math
import torch


@torch.no_grad()
def rescale_adam_row_displacement(parameter, before, rows, multiplier):
    """Apply a row-specific LR to a completed Adam update, before constraints.

    Adam normalizes gradient magnitude, so multiplying selected gradients is
    not a row-specific learning rate. Its displacement *is* linear in LR.
    Moment estimates must remain untouched. The caller runs all ordinary
    optical-mass compensation and ownership projections after this function.
    No temporal state is kept here, so append/prune remaps cannot stale a mask.
    """
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Row LR multiplier must be finite and positive")
    if before.shape != parameter.shape or rows.shape != (parameter.shape[0],):
        raise ValueError("Row LR update state is not aligned")
    if rows.dtype != torch.bool:
        raise ValueError("Row LR selection must be boolean")
    if multiplier == 1. or not bool(rows.any()): return
    parameter[rows] = before[rows] + multiplier*(parameter[rows]-before[rows])
