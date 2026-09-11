import torch
import pytest
from scripts.canopy_native_surface_prefix import ordered_prefix,prefix_rgb,native_prefix,surface_api


def test_surface_prefix_respects_depth_and_id_ties():
    p=ordered_prefix(torch.tensor([3,1,2]),torch.tensor([4.,2.,2.]),
        torch.tensor([.3,.2,.1]),torch.eye(3))
    assert p['ids'].tolist()==[1,2,3]
    torch.testing.assert_close(prefix_rgb(p,torch.tensor([1.,2.,3.,5.])),
        torch.tensor([[0.,0.,0.],[0.,.2,.1],[0.,.2,.1],[.3,.2,.1]]))


def test_empty_prefix_is_zero():
    p=ordered_prefix(torch.empty(0,dtype=torch.long),torch.empty(0),torch.empty(0),torch.empty(0,3))
    assert not prefix_rgb(p,torch.tensor([1.,3.])).any()


@pytest.mark.parametrize('tilted',[False,True])
def test_native_surface_prefix_matches_actual_black_screen(tilted):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import importlib.util
    from pathlib import Path
    _C=surface_api()
    spec=importlib.util.spec_from_file_location('prefix_native_helpers',Path(__file__).with_name('test_native_mixed_rasterizer.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    _,_,Rasterizer=h._api();settings=h._settings()
    xyz=torch.tensor([[0.,0.,2.],[0.,0.,4.],[0.,0.,4.1]],device='cuda')
    scales=torch.full((3,2),10.,device='cuda');quat=torch.tensor([[1.,0.,0.,0.]]*3,device='cuda')
    if tilted:quat[0]=torch.tensor([.94,0.,.341174,0.],device='cuda');quat=quat/quat.norm(dim=1,keepdim=True)
    opacity=torch.tensor([.4,.99,.99],device='cuda');colors=torch.ones(3,3,device='cuda')
    x,y=34,20;fields=torch.zeros(1,47,61,device='cuda');fields[0,y,x]=1
    empty=torch.empty(0,3,device='cuda')
    wall=Rasterizer(settings)(xyz,torch.zeros_like(xyz),scales,quat,empty,empty,empty,
        torch.empty(0,4,device='cuda'),colors,opacity,audit_fields=fields)
    projected=_C.diagnostic_surface_support(xyz,scales,quat,opacity,settings.viewmatrix,settings.projmatrix,61,47)
    p=native_prefix(projected,opacity,colors,wall[3][:,1],(x,y))
    torch.testing.assert_close(prefix_rgb(p,100.),wall[0][:,y,x],atol=2e-6,rtol=2e-6)
    for z in (1.,3.,5.):
        vol=torch.tensor([[0.,0.,z]]*3,device='cuda');vq=torch.tensor([[1.,0.,0.,0.]]*3,device='cuda')
        screen=Rasterizer(settings)(xyz,torch.zeros_like(xyz),scales,quat,
            vol,torch.zeros_like(vol),torch.full((3,3),10.,device='cuda'),vq,
            torch.cat((colors,torch.zeros(3,3,device='cuda'))),
            torch.cat((opacity,torch.full((3,),.999999,device='cuda'))))
        torch.testing.assert_close(prefix_rgb(p,z),screen[0][:,y,x],atol=2e-4,rtol=2e-4)
