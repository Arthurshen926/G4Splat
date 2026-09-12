"""Reversible optimization of existing surfaces; source tensors remain a reference."""
import math
import torch


class SurfaceGeometryDelta(torch.nn.Module):
    def __init__(self,base,*,enabled,maximum_displacement=.5):
        super().__init__()
        if not math.isfinite(maximum_displacement) or maximum_displacement<=0:
            raise ValueError('Positive finite geometric trust radius required')
        self.base=base;self.enabled=bool(enabled);self.maximum_displacement=float(maximum_displacement)
        self.active_sh_degree=base.active_sh_degree;self.max_sh_degree=base.max_sh_degree
        self.position_code=torch.nn.Parameter(torch.zeros_like(base.get_xyz),requires_grad=self.enabled)
        self.scale_code=torch.nn.Parameter(torch.zeros_like(base.get_scaling),requires_grad=self.enabled)
        self.rotation_code=torch.nn.Parameter(torch.zeros_like(base.get_rotation),requires_grad=self.enabled)

    @property
    def get_xyz(self):
        return self.base.get_xyz.detach()+self.position_code.tanh()*(self.maximum_displacement/math.sqrt(3))
    @property
    def get_scaling(self):
        return self.base.get_scaling.detach()*(math.log(2)*self.scale_code.tanh()).exp()
    @property
    def get_rotation(self):
        q=self.base.get_rotation.detach()
        delta=torch.nn.functional.normalize(q+.25*self.rotation_code.tanh(),dim=-1)-torch.nn.functional.normalize(q,dim=-1)
        return q+delta
    @property
    def get_opacity(self):return self.base.get_opacity.detach()
    @property
    def get_features(self):return self.base.get_features.detach()

    def optimization_groups(self,learning_rate=.001):
        if not self.enabled:return []
        if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Positive geometric learning rate required')
        return [dict(params=[self.position_code],lr=learning_rate),dict(params=[self.scale_code],lr=learning_rate*.5),
                dict(params=[self.rotation_code],lr=learning_rate*.5)]

    @torch.no_grad()
    def movement_audit(self):
        distance=(self.get_xyz-self.base.get_xyz).norm(dim=-1)
        return dict(rows=len(distance),moved_rows=int((distance>1e-7).sum()),
                    mean_displacement=float(distance.mean()),maximum_displacement=float(distance.max()),
                    changed_scale_rows=int((self.scale_code.abs().amax(-1)>1e-7).sum()),
                    changed_rotation_rows=int((self.rotation_code.abs().amax(-1)>1e-7).sum()))
