from types import SimpleNamespace
import torch
from outdoor.surface_geometry_delta import SurfaceGeometryDelta
from outdoor.native_detail_suffix import NativeDetailSuffix


def test_zero_identity_geometry_gradients_and_reference_isolation():
    base=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,
        get_xyz=torch.randn(4,3,requires_grad=True),get_scaling=torch.ones(4,2,requires_grad=True),
        get_rotation=torch.tensor([[1.,0.,0.,0.]]*4),get_opacity=torch.ones(4,1),get_features=torch.zeros(4,16,3))
    g=SurfaceGeometryDelta(base,enabled=True)
    for key in ('get_xyz','get_scaling','get_rotation'):assert torch.equal(getattr(g,key),getattr(base,key))
    s=NativeDetailSuffix(g,torch.ones(1,3),torch.tensor([[0.,0.,1.]]),torch.ones(1,3)*.5,torch.ones(1)*.01,train_base_geometry=True)
    (s.get_xyz.sum()+s.get_scaling.sum()).backward()
    assert g.position_code.grad.abs().sum()>0 and g.scale_code.grad.abs().sum()>0
    assert base.get_xyz.grad is None and base.get_scaling.grad is None
    with torch.no_grad():g.position_code.fill_(100)
    assert (g.get_xyz-base.get_xyz).norm(dim=-1).max()<=.500001
