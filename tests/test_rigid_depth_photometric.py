import torch
from scripts.audit_rigid_depth_photometric import matched_neighbor_cost


def test_all_hypotheses_share_neighbor_set():
    x=torch.tensor([[[1.,9.,5.]],[[3.,7.,5.]],[[0.,float('inf'),0.]]])
    cost,count=matched_neighbor_cost(x)
    assert count.tolist()==[2]
    torch.testing.assert_close(cost,torch.tensor([[2.,8.,5.]]))


def test_one_common_neighbor_is_not_enough():
    cost,count=matched_neighbor_cost(torch.tensor([[[1.,2.]],[[float('inf'),1.]]]))
    assert count.tolist()==[1] and torch.isnan(cost).all()
