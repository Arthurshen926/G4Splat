from types import SimpleNamespace
import torch
from outdoor.native_detail_suffix import NativeDetailSuffix


def test_suffix_preserves_prefix_and_only_exposes_own_optics_gradients():
    base=SimpleNamespace(active_sh_degree=3,max_sh_degree=3,
        get_xyz=torch.randn(4,3,requires_grad=True),get_scaling=torch.ones(4,2,requires_grad=True),
        get_rotation=torch.tensor([[1.,0.,0.,0.]]*4,requires_grad=True),
        get_opacity=torch.ones(4,1,requires_grad=True),get_features=torch.randn(4,16,3,requires_grad=True))
    suffix=NativeDetailSuffix(base,torch.randn(2,3),torch.tensor([[0.,0.,1.],[0.,0.,-1.]]),
                              torch.ones(2,3)*.3,torch.ones(2)*.01)
    for attr in ('get_xyz','get_scaling','get_rotation','get_opacity','get_features'):
        assert torch.equal(getattr(suffix,attr)[:4],getattr(base,attr))
    (suffix.get_opacity.sum()+suffix.get_features.sum()).backward()
    assert base.get_opacity.grad is None and base.get_features.grad is None
    assert suffix.logit.grad is not None and suffix.dc.grad is not None
    assert torch.isfinite(suffix.get_rotation).all()
    refined=NativeDetailSuffix(base,torch.randn(2,3),torch.tensor([[0.,0.,1.],[0.,0.,1.]]),
                              torch.ones(2,3)*.3,torch.ones(2)*.01,refine_proposals=True)
    (refined.get_xyz.sum()+refined.get_scaling.sum()).backward()
    assert refined.position_delta.grad is not None and refined.scale_delta.grad is not None
    assert base.get_xyz.grad is None
