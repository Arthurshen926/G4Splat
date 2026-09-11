"""Additive tau coordinates for explicitly selected diagnostic candidates.

Not a thickness target or production change. Paired logit/tau comparisons
use the SAME opacity bounds and preserve rendered logits in checkpoints.
"""
import math
import torch


class CandidateOpticalCoordinate:
    def __init__(self, logits, *, opacity_floor=1e-5, opacity_ceiling=1-1e-6):
        if (not isinstance(logits,torch.nn.Parameter) or not logits.requires_grad
                or not torch.isfinite(logits).all() or not 0<opacity_floor<opacity_ceiling<1):
            raise ValueError('Live finite logits and explicit matching bounds required')
        self.logits=logits
        self.minimum=-math.log1p(-opacity_floor)
        self.maximum=-math.log1p(-opacity_ceiling)
        initial=torch.nn.functional.softplus(logits.detach())
        if (initial<self.minimum).any() or (initial>self.maximum).any():
            raise ValueError('Initialization must already satisfy the shared bounds')
        self.optical_depth=torch.nn.Parameter(initial.clone())

    def clear_gradient(self):
        # The renderer's logits are not in the tau optimizer parameter group.
        self.logits.grad=None

    @torch.no_grad()
    def transfer_gradient(self):
        if self.logits.grad is None or not torch.isfinite(self.logits.grad).all():
            raise ValueError('Finite rendered logit gradient required')
        torch.testing.assert_close(torch.nn.functional.softplus(self.logits),self.optical_depth,rtol=2e-5,atol=1e-8)
        gradient=self.logits.grad/self.logits.sigmoid()
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite optical-coordinate gradient')
        self.optical_depth.grad=gradient.clone()

    @torch.no_grad()
    def project_and_sync(self):
        if not torch.isfinite(self.optical_depth).all():raise ValueError('Nonfinite optical update')
        self.optical_depth.clamp_(self.minimum,self.maximum)
        # Stable inverse softplus; no permanently dead zero-alpha state.
        self.logits.copy_(self.optical_depth+torch.log(-torch.expm1(-self.optical_depth)))

    def state_dict(self):
        return dict(version=1,minimum=self.minimum,maximum=self.maximum,
                    optical_depth=self.optical_depth.detach().cpu().clone())

    @torch.no_grad()
    def load_state_dict(self,state):
        value=state['optical_depth']
        if (state.get('version')!=1 or state['minimum']!=self.minimum or state['maximum']!=self.maximum
                or value.shape!=self.optical_depth.shape or value.dtype!=self.optical_depth.dtype
                or not torch.isfinite(value).all() or (value<self.minimum).any() or (value>self.maximum).any()):
            raise ValueError('Optical coordinate identity/bounds changed')
        torch.testing.assert_close(value.to(self.logits),torch.nn.functional.softplus(self.logits),rtol=2e-5,atol=1e-8)
        self.optical_depth.copy_(value)
