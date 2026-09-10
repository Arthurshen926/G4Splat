"""Explicit persistent-optics exception to the frozen-source candidate adapter."""
import torch
from scripts.canopy_candidate_diagnostic import NativeCandidateView


class JointOpticalCandidateView(NativeCandidateView):
    def __init__(self, base, candidates, eligible):
        if base.dynamic_leaf_mask.any() or base.persistent_envelope_mask.any():
            raise ValueError('Static leaf-only joint optics required')
        if eligible.dtype != torch.bool or eligible.shape != base.static_leaf_mask.shape:
            raise ValueError('Explicit source optical authority mask required')
        if (eligible & ~base.static_leaf_mask).any(): raise ValueError('Only leaf optics may change')
        trainable = {name for name, p in base.named_parameters() if p.requires_grad}
        if trainable != {'features', 'opacity_logits'}:
            raise ValueError('Only source features and opacity may be trainable')
        self.base, self.candidates, self.sh_degree = base, candidates, base.sh_degree
        self.optical_hooks = [parameter.register_hook(
            lambda grad: grad*eligible.reshape((-1,)+(1,)*(grad.ndim-1)))
            for parameter in (base.features, base.opacity_logits)]
