from types import SimpleNamespace
import torch
from scripts.canopy_stratified_depth_candidates import stratified_ray_candidates


def test_stratified_hypotheses_reproject_and_are_reproducible_without_global_rng_change():
    dtype = torch.float64
    rotation = torch.tensor([[.8, 0., .6], [.36, .8, -.48], [-.48, .6, .64]], dtype=dtype)
    center = torch.tensor([31., -12., 7.], dtype=dtype)
    transform = torch.eye(4, dtype=dtype); transform[:3, :3] = rotation; transform[3, :3] = -center@rotation
    view = SimpleNamespace(cx=317.4, cy=181.9, focal_x=713., focal_y=691., world_view_transform=transform, camera_center=center)
    uv = torch.tensor([[15., 30.], [600., 330.], [317.4, 181.9]], dtype=dtype)
    depth = torch.tensor([4., 17., 28.], dtype=dtype); colors = torch.full((3, 3), .4, dtype=dtype)
    rng = torch.random.get_rng_state().clone()
    xyz, scales, result_colors = stratified_ray_candidates(view, uv, depth, colors, seed=7)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(xyz, stratified_ray_candidates(view, uv, depth, colors, seed=7)[0])
    camera = torch.cat((xyz, torch.ones(len(xyz), 1, dtype=dtype)), 1)@transform
    projected = torch.stack((camera[:, 0]/camera[:, 2]*view.focal_x+view.cx,
                             camera[:, 1]/camera[:, 2]*view.focal_y+view.cy), 1)
    torch.testing.assert_close(projected, uv[:, None].expand(-1, 3, -1).reshape(-1, 2), rtol=0, atol=1e-10)
    ratios = camera[:, 2].reshape(-1, 3)/depth[:, None]
    edges = torch.linspace(torch.tensor(.4).double().log(), torch.tensor(1.2).double().log(), 4).exp()
    assert ((ratios > edges[:-1]) & (ratios < edges[1:])).all()
    torch.testing.assert_close(scales[:, 0]/camera[:, 2], torch.full((9,), 1.2/(713*691)**.5, dtype=dtype))
    assert torch.equal(result_colors, colors[:, None].expand(-1, 3, -1).reshape(-1, 3))
