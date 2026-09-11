import math
import torch
import pytest
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_spatial_footprint import SpatialFootprintCandidates
from scripts.canopy_selective_shape import SelectiveSpatialFootprintCandidates,select_conflicting_candidates,initial_geometry_sha256,load_shape_cohort


def cloud():
    return CandidateCloud(torch.zeros(2,3),torch.ones(2,3),torch.zeros(2,3))


def test_zero_shape_matches_isotropic_forward_and_gradient():
    a=SpatialFootprintCandidates(cloud(),8.,math.log(2.))
    b=SelectiveSpatialFootprintCandidates(cloud(),torch.tensor([True,False]),8.,math.log(2.))
    with torch.no_grad():a.scale_code.fill_(.3);b.scale_code.copy_(a.scale_code)
    torch.testing.assert_close(a.scales,b.scales,rtol=0,atol=0)
    a.scales.square().sum().backward();b.scales.square().sum().backward()
    torch.testing.assert_close(a.scale_code.grad,b.scale_code.grad,rtol=0,atol=0)


def test_shape_never_expands_axis_or_position_bounds_and_excluded_rows():
    b=SelectiveSpatialFootprintCandidates(cloud(),torch.tensor([True,False]),8.,math.log(2.))
    xyz=b.xyz.clone();identity=initial_geometry_sha256(b.cloud)
    with torch.no_grad():b.shape_code.copy_(torch.tensor([[100.,-100.,0.],[100.,-100.,0.]]))
    assert b.scales.min()>=.5 and b.scales.max()<=2
    torch.testing.assert_close(b.scales[1],torch.ones(3),rtol=0,atol=0)
    torch.testing.assert_close(b.xyz,xyz,rtol=0,atol=0)
    assert initial_geometry_sha256(b.cloud)==identity


def test_selection_needs_multiview_interior_rigid_evidence():
    mass=torch.tensor([[1.,.1,0.],[1.,0.,.1],[1.,.0001,0.]])
    counts=torch.tensor([[2,2,0],[2,0,10],[2,2,0]])
    assert select_conflicting_candidates(mass,counts).tolist()==[True,False,False]


def test_cohort_binds_initial_geometry_and_training_views(tmp_path):
    c=cloud();mass=torch.tensor([[1.,.1,0.],[1.,0.,0.]])
    counts=torch.tensor([[2,2,0],[2,0,0]])
    eligible=select_conflicting_candidates(mass,counts)
    path=tmp_path/'cohort.pth'
    torch.save(dict(eligible=eligible,mass=mass,counts=counts,report=dict(
        source_checkpoint_sha256='source',initial_geometry_sha256=initial_geometry_sha256(c),
        training_views=[1,2],excluded_views=[3],selected_count=1)),path)
    assert load_shape_cohort(path,c,'source',[1,2],[3]).tolist()==[True,False]
    with pytest.raises(ValueError):load_shape_cohort(path,c,'source',[1,3],[2])
    c.xyz[0,0]=.01
    with pytest.raises(ValueError):load_shape_cohort(path,c,'source',[1,2],[3])


def test_native_shape_derivative_matches_finite_difference():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('native_shape_helpers',Path(__file__).with_name('test_native_mixed_rasterizer.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    _,_,Rasterizer=helpers._api();settings=helpers._settings()
    xyz=torch.tensor([[.07,0.,2.]],device='cuda');empty=torch.empty(0,3,device='cuda')
    c=CandidateCloud(xyz,torch.full_like(xyz,.2),torch.zeros_like(xyz))
    b=SelectiveSpatialFootprintCandidates(c,torch.tensor([True],device='cuda'),8.,math.log(2.))
    def value():
        rgb=Rasterizer(settings)(empty,empty,torch.empty(0,2,device='cuda'),torch.empty(0,4,device='cuda'),
            b.xyz,torch.zeros_like(xyz),b.scales,b.quaternions,torch.ones(1,3,device='cuda'),
            torch.full((1,),.7,device='cuda'))[0]
        return rgb[:,18:22,30:34].sum()
    value().backward();gradient=b.shape_code.grad[0,0].clone()
    with torch.no_grad():
        b.shape_code[0,0]=.001;plus=value();b.shape_code[0,0]=-.001;minus=value()
    assert gradient.abs()>1e-5
    torch.testing.assert_close(gradient,(plus-minus)/.002,atol=.003,rtol=.005)
