import torch
from outdoor.canopy_support_expansion import clip_canopy_depth_queries_before_rigid


def test_hidden_leaf_is_neither_positive_hit_nor_rigid_free_space():
    # Wall z=5, monocular background z=10: a hidden leaf z=7 was formerly
    # inside the intrinsic pre-hit query (0,8), although the wall occludes it.
    pre=torch.tensor([[[8.]],[[10.]]]);hit=torch.tensor([[[9.6]],[[10.4]]])
    a,b,v,audit=clip_canopy_depth_queries_before_rigid(pre,hit,torch.ones(1,1,dtype=torch.bool),
        torch.zeros(1,1),torch.ones(1,1),torch.full((1,1),5.),torch.ones(1,1))
    assert v.all() and a[0,0,0]==4.95
    assert torch.isnan(b).all()
    assert 7.<pre[0,0,0] and not 7.<a[0,0,0]
    assert audit['prehit_bounds_clipped_pixels']==1
    assert torch.equal(pre,torch.tensor([[[8.]],[[10.]]]))


def test_conflicting_canopy_background_depth_does_not_erase_foreground():
    pre=torch.tensor([[[8.]],[[10.]]]);hit=torch.tensor([[[9.6]],[[10.4]]])
    a,b,v,audit=clip_canopy_depth_queries_before_rigid(pre,hit,torch.ones(1,1,dtype=torch.bool),
        torch.ones(1,1),torch.zeros(1,1),torch.full((1,1),5.),torch.ones(1,1))
    assert not v.any() and torch.isnan(a).all() and torch.isnan(b).all()
    assert audit['canopy_depth_conflict_unknown_pixels']==1


def test_front_hit_is_clipped_at_wall_and_unknown_wall_does_not_change_queries():
    pre=torch.tensor([[[3.,3.]],[[4.8,4.8]]]);hit=torch.tensor([[[4.6,4.6]],[[5.1,5.1]]])
    a,b,v,_=clip_canopy_depth_queries_before_rigid(pre,hit,torch.ones(1,2,dtype=torch.bool),
        torch.ones(1,2),torch.zeros(1,2),torch.full((1,2),5.),torch.tensor([[1.,.4]]))
    assert torch.equal(a,pre) and v.all()
    assert b[1,0,0]==4.95 and torch.equal(b[:,:,1],hit[:,:,1])
