"""Explicit bounded static-position exception for supported source leaves only."""
import math
import torch


class StaticSourcePosition(torch.nn.Module):
    def __init__(self, xyz, scales, eligible, radius_sigmas):
        super().__init__()
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape != scales.shape
                or eligible.dtype != torch.bool or eligible.shape != xyz.shape[:1]
                or not torch.isfinite(xyz).all() or not torch.isfinite(scales).all()
                or not (scales > 0).all() or not eligible.any()
                or not math.isfinite(radius_sigmas) or not 0 < radius_sigmas <= 4):
            raise ValueError('Finite geometry, explicit supported leaves and bounded radius required')
        self.radius_sigmas = float(radius_sigmas)
        self.row_count = len(xyz)
        self.register_buffer('indices', eligible.nonzero().flatten().detach().clone())
        # Euclidean displacement is bounded by radius * INITIAL largest sigma.
        # No learned size, rotation or depth target enters this bound.
        self.register_buffer('extent', scales[eligible].amax(1, keepdim=True).detach().clone())
        self.code = torch.nn.Parameter(torch.zeros_like(xyz[eligible]))

    def delta(self):
        return self.radius_sigmas*self.extent*self.code/torch.sqrt(1+self.code.square().sum(1, keepdim=True))

    def apply(self, xyz):
        if xyz.shape != (self.row_count, 3):
            raise ValueError('Source row identity must remain unchanged')
        return xyz.index_add(0, self.indices, self.delta())

    def load_verified_state(self, state):
        if (set(state) != {'indices', 'extent', 'code'}
                or state['indices'].dtype != self.indices.dtype or state['extent'].dtype != self.extent.dtype
                or state['code'].dtype != self.code.dtype
                or not torch.equal(state['indices'].to(self.indices), self.indices)
                or not torch.equal(state['extent'].to(self.extent), self.extent)
                or state['code'].shape != self.code.shape or not torch.isfinite(state['code']).all()):
            raise ValueError('Checkpoint cannot broaden source-position authority or bounds')
        self.load_state_dict(state)

    @torch.no_grad()
    def audit(self):
        delta = self.delta()
        return dict(eligible_rows=len(self.indices), moved_rows=int((delta != 0).any(1).sum()),
                    maximum_displacement=float(delta.norm(dim=1).max()),
                    maximum_bound_fraction=float((delta.norm(dim=1)/self.extent[:, 0]/self.radius_sigmas).max()),
                    finite=bool(torch.isfinite(delta).all()), static_unconditioned=True)


class StaticSourcePositionView:
    """Preserve source appearance, evidence and scales; change only static XYZ."""
    def __init__(self, base, position):
        if (base.dynamic_leaf_mask.any() or base.persistent_envelope_mask.any()
                or (~base.static_leaf_mask[position.indices]).any()):
            raise ValueError('Only static leaf geometry may receive this exception')
        self.base, self.position = base, position

    def __len__(self): return len(self.base)
    def __getattr__(self, name): return getattr(self.base, name)
    @property
    def xyz(self): return self.position.apply(self.base.xyz)

    def conditioned_state(self, temporal_code, *, include_dynamic, **kwargs):
        if temporal_code is not None or include_dynamic:
            raise ValueError('Source-position control is static and unconditioned')
        if any(value is not None for name, value in kwargs.items() if name != 'conditioned_visibility_gate'):
            raise ValueError('Source-position control does not accept extra gradient authority overrides')
        xyz, features, opacity = self.base.conditioned_state(None, include_dynamic=False, **kwargs)
        return self.position.apply(xyz), features, opacity
