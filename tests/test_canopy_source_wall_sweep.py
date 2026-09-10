import pytest
import torch
from scripts.refine_canopy_ray_depth_consensus import source_wall_depth_factors


def test_unmeasured_depth_hint_does_not_limit_finite_wall_search():
    initial=torch.tensor([8.,2.,5.])
    upper=torch.tensor([10.,10.,float('inf')])
    shifts=torch.linspace(-.3,.3,33)
    factors,bounded=source_wall_depth_factors(initial,upper,shifts,.01)
    depth=initial[:,None]*factors
    assert bounded.tolist()==[True,True,False]
    assert torch.allclose(depth[0],depth[1])
    assert depth[0,0]<.011 and 9.99<depth[0,-1]<10
    assert (depth[:2,1:]>depth[:2,:-1]).all()
    assert torch.equal(factors[2],shifts.exp())


def test_invalid_or_missing_wall_never_creates_an_invented_global_bound():
    shifts=torch.linspace(-.3,.3,5)
    factors,bounded=source_wall_depth_factors(torch.ones(3),torch.tensor([.001,float('nan'),float('inf')]),shifts,.01)
    assert not bounded.any()
    assert torch.equal(factors,shifts.exp()[None].expand(3,-1))
    with pytest.raises(ValueError):source_wall_depth_factors(torch.tensor([0.]),torch.tensor([4.]),shifts,.01)
