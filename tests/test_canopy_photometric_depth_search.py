from types import SimpleNamespace
import torch
from scripts.canopy_photometric_depth_search import photometric_depth_cost
from scripts.canopy_photometric_depth_search import select_depth_hypotheses
from scripts.canopy_photometric_depth_search import refine_stratified_proposals
from scripts.canopy_stratified_depth_candidates import stratified_ray_candidates


def camera(x):
    return SimpleNamespace(cx=32., cy=32., focal_x=100., focal_y=100.,
        camera_center=torch.tensor([x, 0., 0.]), world_view_transform=torch.eye(4))


def test_exact_k_parallax_prefers_correct_depth_without_pseudo_truth():
    rgb = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(22))
    mask = torch.ones(64, 64, dtype=torch.bool)
    # A plane at z2: a .1-unit translated camera sees texture shifted by5px.
    neighbors = [(camera(.1), torch.roll(rgb, -5, 2), mask),
                 (camera(-.1), torch.roll(rgb, 5, 2), mask)]
    costs, support = photometric_depth_cost(camera(0.), torch.tensor([[32., 32.], [24., 28.]]),
        torch.tensor([[1., 2., 4.], [1., 2., 4.]]), rgb, neighbors)
    assert torch.equal(costs.argmin(1), torch.ones(2, dtype=torch.long))
    assert (costs[:, 1] < 1e-5).all() and (support == 2).all()


def test_non_tree_neighbor_is_unsupported_not_valid_depth_evidence():
    rgb = torch.ones(3, 64, 64)*.3
    mask = torch.zeros(64, 64, dtype=torch.bool)
    costs, support = photometric_depth_cost(camera(0.), torch.tensor([[32., 32.]]),
        torch.tensor([[1., 2.]]), rgb, [(camera(.1), rgb, mask), (camera(-.1), rgb, mask)])
    assert torch.isinf(costs).all() and (support == 0).all()


def test_ambiguous_depths_keep_original_hypotheses_exact():
    depth = torch.arange(1., 25.).reshape(1, 3, 8)
    fallback = torch.tensor([[2.5, 10.5, 18.5]])
    costs = torch.zeros_like(depth)
    selected, accepted = select_depth_hypotheses(depth, costs, fallback)
    assert torch.equal(selected, fallback) and not accepted.any()
    costs.fill_(.5); costs[0, 1, 3] = .1
    selected, accepted = select_depth_hypotheses(depth, costs, fallback)
    assert accepted.tolist() == [[False, True, False]]
    assert selected.tolist() == [[2.5, 12., 18.5]]


def test_no_parallax_does_not_provide_depth_evidence():
    rgb = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(22))
    mask = torch.ones(64, 64, dtype=torch.bool)
    costs, support = photometric_depth_cost(camera(0.), torch.tensor([[32., 32.]]),
        torch.tensor([[1., 2., 4.]]), rgb, [(camera(0.), rgb, mask), (camera(0.), rgb, mask)])
    assert torch.isinf(costs).all() and (support == 0).all()


def test_uninformative_search_keeps_candidate_geometry_and_color_bitwise():
    source = camera(0.); rgb = torch.ones(3, 64, 64)*.3
    uv = torch.tensor([[32., 32.]]); depth = torch.tensor([2.])
    proposals = stratified_ray_candidates(source, uv, depth, torch.full((1, 3), .3), seed=9)
    mask = torch.ones(64, 64, dtype=torch.bool)
    refined, audit = refine_stratified_proposals(source, uv, depth, rgb,
        [(camera(.1), rgb, mask), (camera(-.1), rgb, mask)], proposals)
    assert audit['accepted_depth_slots'] == 0
    assert all(torch.equal(a, b) for a, b in zip(refined, proposals))


def test_photometric_projection_is_invariant_to_world_rotation_and_translation():
    rgb = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(22))
    mask = torch.ones(64, 64, dtype=torch.bool)
    source = camera(0.); neighbors = [(camera(.1), torch.roll(rgb, -5, 2), mask),
                                    (camera(-.1), torch.roll(rgb, 5, 2), mask)]
    uv = torch.tensor([[32., 32.]]); depths = torch.tensor([[1., 2., 4.]])
    before, _ = photometric_depth_cost(source, uv, depths, rgb, neighbors)
    rotation = torch.tensor([[.8, 0., .6], [0., 1., 0.], [-.6, 0., .8]])
    translation = torch.tensor([2., -1., 3.])
    for view in [source]+[n[0] for n in neighbors]:
        view.camera_center = view.camera_center@rotation.T+translation
        view.world_view_transform[:3, :3] = rotation
    after, _ = photometric_depth_cost(source, uv, depths, rgb, neighbors)
    torch.testing.assert_close(before, after, atol=5e-5, rtol=1e-4)


def test_source_patch_padding_cannot_fabricate_matching_evidence():
    rgb = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(22))
    mask = torch.ones(64, 64, dtype=torch.bool)
    costs, support = photometric_depth_cost(camera(0.), torch.tensor([[0., 32.]]),
        torch.tensor([[1., 2., 4.]]), rgb, [(camera(-.1), rgb, mask), (camera(-.2), rgb, mask)])
    assert torch.isinf(costs).all() and (support == 0).all()
