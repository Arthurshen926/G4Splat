"""Forward-identical local gradient for tau*area preserving scale updates.

This helper alone does not change production training. It expresses the local
Jacobian of the post-step mass retraction, leaving opacity's own gradient intact.
"""
import torch


def mass_preserving_scale_alpha(opacity, area):
    if opacity.numel() != area.numel():
        raise ValueError('One optical area per opacity required')
    if not torch.is_grad_enabled() or not area.requires_grad:
        return opacity
    area = area.reshape_as(opacity).clamp_min(1e-12)
    tau = -torch.log1p(-opacity.detach().clamp(0., 1.-1e-6))
    fixed_mass_alpha = -torch.expm1(-tau*area.detach()/area)
    # Only the scale derivative is added. Opacity gradients (including their
    # authority gates) remain untouched; the forward alpha is bit-identical.
    return opacity+(fixed_mass_alpha-fixed_mass_alpha.detach())
