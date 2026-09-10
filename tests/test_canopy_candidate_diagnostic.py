from types import SimpleNamespace
import pytest
import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud, NativeCandidateView, ray_candidates


def test_depth_hypotheses_preserve_angular_size_and_colors():
    view = SimpleNamespace(cx=0., cy=0., focal_x=100., focal_y=100.,
                           world_view_transform=torch.eye(4), camera_center=torch.zeros(3))
    xyz, scale, color = ray_candidates(view, torch.tensor([[10., 20.]]), torch.tensor([10.]),
                                       torch.tensor([[.1, .2, .3]]), [.65, .8, 1.])
    torch.testing.assert_close(xyz[:, 2], torch.tensor([6.5, 8., 10.]))
    torch.testing.assert_close(xyz[:, :2]/xyz[:, 2:], torch.tensor([[.1, .2]]).expand(3, 2))
    torch.testing.assert_close(scale[:, 0]/xyz[:, 2], torch.full((3,), .012))
    torch.testing.assert_close(color, torch.tensor([[.1, .2, .3]]).expand(3, 3))


def test_candidates_only_optics_are_trainable_and_no_opacity_floor():
    c = CandidateCloud(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3)*.5)
    assert set(dict(c.named_parameters())) == {'dc', 'logits'}
    torch.testing.assert_close(c.logits.sigmoid(), torch.full((2,), .001))
    opt = torch.optim.Adam(c.parameters(), lr=.1)
    c.logits.sigmoid().sum().backward(); opt.step()
    assert (c.logits.sigmoid() < .001).all()
    with pytest.raises(ValueError): CandidateCloud(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3), .1)


def test_depth_candidates_reproject_with_rotated_translated_exact_k_camera():
    dtype = torch.float64
    rotation = torch.tensor([[.8, 0., .6], [.36, .8, -.48], [-.48, .6, .64]], dtype=dtype)
    center = torch.tensor([31., -12., 7.], dtype=dtype)
    world_view = torch.eye(4, dtype=dtype)
    world_view[:3, :3] = rotation
    world_view[3, :3] = -center@rotation
    view = SimpleNamespace(cx=317.4, cy=181.9, focal_x=713., focal_y=691.,
                           world_view_transform=world_view, camera_center=center)
    uv = torch.tensor([[15., 30.], [600., 330.], [317.4, 181.9]], dtype=dtype)
    depth = torch.tensor([4., 17., 28.], dtype=dtype)
    xyz, _, _ = ray_candidates(view, uv, depth, torch.full((3, 3), .4, dtype=dtype), [.65, .8, 1.])
    camera = torch.cat((xyz, torch.ones(len(xyz), 1, dtype=dtype)), 1)@world_view
    projected = torch.stack((camera[:, 0]/camera[:, 2]*view.focal_x+view.cx,
                             camera[:, 1]/camera[:, 2]*view.focal_y+view.cy), 1)
    torch.testing.assert_close(projected, uv[:, None].expand(-1, 3, -1).reshape(-1, 2), rtol=0, atol=1e-10)


def test_adapter_does_not_fabricate_evidence():
    base = SimpleNamespace(dynamic_leaf_mask=torch.zeros(1, dtype=torch.bool),
                           persistent_envelope_mask=torch.zeros(1, dtype=torch.bool),
                           parameters=lambda: [], sh_degree=3)
    c = CandidateCloud(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3)*.5)
    adapter = NativeCandidateView(base, c)
    with pytest.raises(AttributeError): _ = adapter.verified_camera_count
    with pytest.raises(AttributeError): _ = adapter.verification_state
    with pytest.raises(ValueError): adapter.conditioned_state(None, include_dynamic=True)
