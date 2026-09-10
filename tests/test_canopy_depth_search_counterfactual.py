from types import SimpleNamespace
import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_stratified_depth_candidates import stratified_ray_candidates
from scripts.canopy_depth_search_counterfactual import reseed_stratified_depth_


def test_counterfactual_equals_fresh_reseed_without_changing_optics():
    v = SimpleNamespace(cx=0., cy=0., focal_x=100., focal_y=100.,
                        world_view_transform=torch.eye(4), camera_center=torch.zeros(3))
    uv = torch.tensor([[10., 20.], [30., 15.]]); depth = torch.tensor([10., 20.]); colors = torch.full((2, 3), .3)
    cloud = CandidateCloud(*stratified_ray_candidates(v, uv, depth, colors, seed=37001, low=.4))
    dc, logits = cloud.dc.clone(), cloud.logits.clone()
    reseed_stratified_depth_(cloud, [v], [dict(index=0, rays=2)], old_low=.4, new_low=.2, high=1.2)
    xyz, scales, _ = stratified_ray_candidates(v, uv, depth, colors, seed=37001, low=.2)
    torch.testing.assert_close(cloud.xyz, xyz); torch.testing.assert_close(cloud.scales, scales)
    assert torch.equal(dc, cloud.dc) and torch.equal(logits, cloud.logits)
