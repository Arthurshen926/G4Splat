import torch
from scripts.audit_moge_canopy_occluder_consensus import front_interval_attribution


def test_front_interval_partition_distinguishes_crossing_from_behind():
    depth = torch.tensor([1., 2., 3., 4., float('nan'), 2.])
    bounds = torch.stack((torch.tensor([.5, 1., 1., 3., 1., 1.]),
                          torch.tensor([1.5, 3., 4., 5., 3., 3.])))
    eligible = torch.tensor([True, True, True, True, True, False])
    result = front_interval_attribution(depth,bounds,torch.full_like(depth,2.5),eligible,
                                        torch.tensor([.9,.1,.2,.3,.1,.1]))
    assert all(row['pixels'] == 1 for row in result.values())
    assert result['full_interval_before']['native_alpha_below_0_5'] == 0
    assert result['center_before_interval_crosses']['native_alpha_below_0_5'] == 1


def test_front_interval_touching_boundary_is_not_strictly_in_front():
    depth=torch.tensor([1.,2.])
    result=front_interval_attribution(depth,torch.tensor([[.5,2.],[2.,3.]]),
        torch.tensor([2.,2.]),torch.ones(2,dtype=torch.bool),torch.zeros(2))
    assert result['center_before_interval_crosses']['pixels']==1
    assert result['full_interval_behind']['pixels']==1
