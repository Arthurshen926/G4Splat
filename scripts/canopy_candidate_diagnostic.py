"""Unverified, static RGB candidates for an isolated native-render experiment.

This adapter is intentionally not a production foliage checkpoint. It never
creates verification metadata or changes the source model. Candidates are
jointly depth-sorted with the unchanged scene, not composited as an overlay.
"""
import math
import torch


def ray_candidates(view, uv, depth, colors, ratios, pixel_sigma=1.2):
    ratios = torch.as_tensor(ratios, device=depth.device, dtype=depth.dtype)
    if (ratios.ndim != 1 or not len(ratios) or not torch.isfinite(ratios).all()
            or not ((ratios >= .4) & (ratios <= 1.5)).all()):
        raise ValueError('Finite bounded depth hypotheses required')
    if not math.isfinite(pixel_sigma) or pixel_sigma <= 0:
        raise ValueError('Positive angular footprint required')
    if (uv.shape != (len(depth), 2) or colors.shape != (len(depth), 3)
            or not torch.isfinite(depth).all() or not (depth > 0).all()):
        raise ValueError('Aligned positive ray depth required')
    rays = torch.stack(((uv[:, 0]-view.cx)/view.focal_x,
                        (uv[:, 1]-view.cy)/view.focal_y, torch.ones_like(depth)), 1)
    z = depth[:, None]*ratios[None]
    xyz = (rays[:, None]*z[:, :, None]) @ view.world_view_transform[:3, :3].T + view.camera_center
    scales = z.reshape(-1, 1).expand(-1, 3)*pixel_sigma/math.sqrt(view.focal_x*view.focal_y)
    return xyz.reshape(-1, 3), scales, colors[:, None].expand(-1, len(ratios), -1).reshape(-1, 3)


class CandidateCloud(torch.nn.Module):
    def __init__(self, xyz, scales, colors, opacity=.001):
        super().__init__()
        if not 0 < opacity <= .005:
            raise ValueError('Only low-opacity initialization is authorized')
        if xyz.shape != scales.shape or xyz.shape != colors.shape or xyz.shape[1] != 3:
            raise ValueError('Aligned candidate tensors required')
        if not all(torch.isfinite(t).all() for t in (xyz, scales, colors)) or not (scales > 0).all():
            raise ValueError('Finite candidates required')
        self.register_buffer('xyz', xyz.detach().clone())
        self.register_buffer('scales', scales.detach().clone())
        q = torch.zeros((len(xyz), 4), device=xyz.device, dtype=xyz.dtype); q[:, 0] = 1
        self.register_buffer('quaternions', q)
        self.dc = torch.nn.Parameter((colors.detach().clamp(0, 1)-.5)/.28209479177387814)
        self.logits = torch.nn.Parameter(torch.full_like(xyz[:, 0], math.log(opacity/(1-opacity))))


class NativeCandidateView:
    """Narrow adapter, supporting only a frozen, static, leaf-only source."""
    def __init__(self, base, candidates):
        if bool(base.dynamic_leaf_mask.any()) or bool(base.persistent_envelope_mask.any()):
            raise ValueError('This diagnostic supports static leaf-only sources')
        if any(p.requires_grad for p in base.parameters()):
            raise ValueError('All existing foliage must remain frozen')
        self.base, self.candidates = base, candidates
        self.sh_degree = base.sh_degree

    def __len__(self):
        return len(self.base)+len(self.candidates.xyz)

    @property
    def scales(self):
        return torch.cat((self.base.scales, self.candidates.scales))

    @property
    def normalized_quaternions(self):
        return torch.cat((self.base.normalized_quaternions, self.candidates.quaternions))

    def __getattr__(self, name):
        # These are renderer shape/role fields, NOT existence/support evidence.
        fills = {'dynamic_leaf_mask': False, 'persistent_envelope_mask': False,
                 'static_leaf_mask': True, 'detail_leaf_mask': True,
                 'replacement_group': -1, 'initialization_source': -1}
        if name not in fills:
            raise AttributeError(f'Candidate diagnostic must not fabricate {name}')
        value = getattr(self.base, name)
        return torch.cat((value, torch.full((len(self.candidates.xyz),), fills[name],
                                           device=value.device, dtype=value.dtype)))

    def conditioned_state(self, temporal_code, *, include_dynamic, **kwargs):
        if temporal_code is not None or include_dynamic:
            raise ValueError('Candidate evaluation must be static and unconditioned')
        base_kwargs = {}
        for key, value in kwargs.items():
            if value is not None:
                if key != 'conditioned_visibility_gate':
                    raise ValueError('No candidate gradient authority overrides allowed')
                value = value[:len(self.base)]
            base_kwargs[key] = value
        xyz, features, opacity = self.base.conditioned_state(None, include_dynamic=False, **base_kwargs)
        extra = features.new_zeros((len(self.candidates.xyz), features.shape[1]-1, 3))
        candidate_features = torch.cat((self.candidates.dc[:, None], extra), 1)
        return (torch.cat((xyz, self.candidates.xyz)), torch.cat((features, candidate_features)),
                torch.cat((opacity, self.candidates.logits.sigmoid())))
