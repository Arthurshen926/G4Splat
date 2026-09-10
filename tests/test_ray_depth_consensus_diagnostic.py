import torch
from scripts.refine_canopy_ray_depth_consensus import select_single_depth


def test_one_depth_per_ray_requires_two_views_and_prefers_no_free_space_conflict():
    positive=torch.tensor([[3,5,6],[1,1,1],[2,2,2],[9,9,9]])
    negative=torch.tensor([[0,1,2],[0,0,0],[0,0,0],[2,3,4]])
    chosen,valid=select_single_depth(positive,negative,torch.tensor([-.15,0.,.15]))
    assert chosen[0]==0 and chosen[2]==1
    assert valid.tolist()==[True,False,True,False]


def test_source_observed_occluder_cannot_be_selected_behind_its_known_wall():
    positive=torch.tensor([[9,3,2],[9,8,7]])
    chosen,valid=select_single_depth(positive,torch.zeros_like(positive),torch.tensor([-.1,0.,.1]),
        torch.tensor([[False,True,True],[False,False,False]]))
    assert chosen[0]==1 and valid.tolist()==[True,False]


def test_visual_hull_nominates_one_front_hit_not_maximum_interior_occupancy():
    positive=torch.tensor([[2,10,20],[2,4,9]])
    negative=torch.tensor([[0,0,0],[1,0,0]])
    chosen,valid=select_single_depth(positive,negative,torch.tensor([-.3,0.,.3]),frontmost=True)
    assert chosen.tolist()==[0,1] and valid.all()
