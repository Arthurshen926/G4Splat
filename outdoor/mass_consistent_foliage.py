"""Training-only local mass coordinates; checkpoint and render values unchanged."""
import torch
from outdoor.hybrid_gaussian_renderer import (
    VolumetricFoliageModel, _identity_with_gradient_gate,
    projected_gaussian_cross_section,
)
from outdoor.optical_mass_gradient import mass_preserving_scale_alpha


class IdentityPreservingMassFoliage(VolumetricFoliageModel):
    @torch.no_grad()
    def restore_integrated_optical_mass(self, target_mass, *, minimum_opacity=1e-6,
                                        maximum_opacity=1.-1e-6, rows=None):
        logits_before = self.opacity_logits.detach().clone()
        opacity_before = self.opacities.detach()
        target = torch.as_tensor(target_mass, device=self.xyz.device, dtype=self.xyz.dtype).reshape(-1)
        if target.shape != (len(self),): raise ValueError('Target mass must match foliage rows')
        unchanged = ((target == self.integrated_optical_mass())
                     & (opacity_before >= minimum_opacity) & (opacity_before <= maximum_opacity))
        if rows is not None:
            if rows.shape != (len(self),) or rows.dtype != torch.bool:
                raise ValueError('Explicit row mask required')
            unchanged |= ~rows
        super().restore_integrated_optical_mass(target, minimum_opacity=minimum_opacity,
                                                maximum_opacity=maximum_opacity)
        self.opacity_logits[unchanged] = logits_before[unchanged]
        relative = (self.opacities-opacity_before).abs()/opacity_before.clamp_min(minimum_opacity)
        return dict(target_mass=float(target.sum()), realized_mass=float(self.integrated_optical_mass().sum()),
                    mean_relative_alpha_compensation=float(relative.mean()) if len(relative) else 0.,
                    maximum_relative_alpha_compensation=float(relative.max()) if len(relative) else 0.,
                    exact_noop_rows=int(unchanged.sum()))


class MassConsistentFoliage(IdentityPreservingMassFoliage):
    def integrated_optical_mass(self):
        # In these local coordinates geometry preserves mass exactly. Keep
        # direct opacity priors on the base getter: they must not acquire
        # ungated geometry authority merely by referencing peak alpha.
        area = projected_gaussian_cross_section(self.scales).detach()
        tau = -torch.log1p(-self.opacities.clamp(0., 1.-1e-6))
        return tau*area

    def conditioned_state(self, temporal_code, *, include_dynamic, **kwargs):
        if bool(self.dynamic_leaf_mask.any()):
            raise ValueError('Mass-consistent scale gradients currently require static foliage')
        xyz, features, opacity = super().conditioned_state(
            temporal_code, include_dynamic=include_dynamic, **kwargs)
        if not torch.is_grad_enabled() or not self.log_scales.requires_grad:
            return xyz, features, opacity
        scales = _identity_with_gradient_gate(
            self.scales, kwargs.get('base_geometry_gradient_gate'))
        area = projected_gaussian_cross_section(scales)
        return xyz, features, mass_preserving_scale_alpha(opacity, area)
