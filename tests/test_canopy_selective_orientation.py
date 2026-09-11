import math
import pytest
import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_selective_orientation import OrientedSelectiveCandidates


def model(device='cpu'):
    xyz=torch.tensor([[.07,0.,2.],[0.,0.,3.]],device=device)
    cloud=CandidateCloud(xyz,torch.full_like(xyz,.2),torch.zeros_like(xyz))
    return OrientedSelectiveCandidates(cloud,torch.tensor([True,False],device=device),8.,math.log(2.))


def test_identity_orientation_and_finite_zero_derivative():
    m=model();torch.testing.assert_close(m.quaternions,m.cloud.quaternions,rtol=0,atol=0)
    m.quaternions[:,1:].sum().backward()
    assert torch.isfinite(m.rotation_code.grad).all()
    assert not m.rotation_code.grad.any()
    with torch.no_grad():m.shape_code[0]=torch.tensor([.1,0.,-.1])
    m.zero_grad(set_to_none=True)
    m.quaternions[:,1:].sum().backward()
    assert m.rotation_code.grad[0].abs().sum()>0
    assert not m.rotation_code.grad[1].any()


def test_orientation_bounded_and_ineligible_unchanged():
    m=model()
    with torch.no_grad():m.shape_code[0]=torch.tensor([.1,0.,-.1])
    xyz=m.xyz.clone();scales=m.scales.clone()
    with torch.no_grad():m.rotation_code.fill_(1000.)
    q=m.quaternions
    torch.testing.assert_close(q.norm(dim=1),torch.ones(2))
    torch.testing.assert_close(q[1],m.cloud.quaternions[1],rtol=0,atol=0)
    assert (2*q[0,0].acos())<=math.pi/2+1e-6
    torch.testing.assert_close(m.xyz,xyz,rtol=0,atol=0)
    torch.testing.assert_close(m.scales,scales,rtol=0,atol=0)


def test_native_isotropic_rotation_cannot_amplify_roundoff_into_adam_step():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('isotropic_native_helpers',Path(__file__).with_name('test_native_mixed_rasterizer.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    _,_,Rasterizer=helpers._api();settings=helpers._settings();m=model('cuda')
    empty=torch.empty(0,3,device='cuda')
    optimizer=torch.optim.Adam([m.rotation_code],lr=.02,eps=1e-15)
    for code in (0.,.2):
        with torch.no_grad():m.rotation_code[0].fill_(code)
        optimizer.zero_grad(set_to_none=True)
        before=m.rotation_code.detach().clone()
        rgb=Rasterizer(settings)(empty,empty,torch.empty(0,2,device='cuda'),torch.empty(0,4,device='cuda'),
            m.xyz,torch.zeros_like(m.xyz),m.scales,m.quaternions,torch.ones(2,3,device='cuda'),
            torch.tensor([.001,0.],device='cuda'))[0]
        rgb[:,18:22,30:34].sum().backward()
        assert not m.rotation_code.grad.any()
        optimizer.step()
        torch.testing.assert_close(m.rotation_code,before,rtol=0,atol=0)


def test_native_orientation_gradient_matches_finite_difference():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('orientation_native_helpers',Path(__file__).with_name('test_native_mixed_rasterizer.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    _,_,Rasterizer=helpers._api();settings=helpers._settings();m=model('cuda')
    with torch.no_grad():
        m.shape_code[0]=torch.tensor([.8,-.8,0.],device='cuda')
        m.rotation_code[0]=torch.tensor([.1,.2,.3],device='cuda')
    empty=torch.empty(0,3,device='cuda')
    def value():
        rgb=Rasterizer(settings)(empty,empty,torch.empty(0,2,device='cuda'),torch.empty(0,4,device='cuda'),
            m.xyz,torch.zeros_like(m.xyz),m.scales,m.quaternions,torch.ones(2,3,device='cuda'),
            torch.tensor([.7,0.],device='cuda'))[0]
        return rgb[:,18:22,30:34].sum()
    value().backward();gradient=m.rotation_code.grad[0,2].clone()
    with torch.no_grad():
        m.rotation_code[0,2]=.301;plus=value();m.rotation_code[0,2]=.299;minus=value()
    assert gradient.abs()>1e-5
    torch.testing.assert_close(gradient,(plus-minus)/.002,atol=.004,rtol=.008)
