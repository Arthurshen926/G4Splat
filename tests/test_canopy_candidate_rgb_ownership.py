import torch
from scripts.canopy_candidate_rgb_ownership import candidate_rgb_loss


def test_background_residual_cannot_fund_wrong_tree_opacity():
    phi = torch.tensor(-4., requires_grad=True)
    reference = torch.full((3, 1, 2), .8, requires_grad=True)
    target = torch.full((3, 1, 2), .3)
    rgb = reference.detach()*(1-phi.sigmoid())+.1*phi.sigmoid()
    regions = {'tree': torch.zeros(1, 2, dtype=torch.bool),
               'rigid': torch.tensor([[True, False]]), 'sky': torch.tensor([[False, True]])}
    legacy = candidate_rgb_loss(rgb, target, reference, regions, noncanopy_target='legacy_ground_truth')
    safe = candidate_rgb_loss(rgb, target, reference, regions, noncanopy_target='source_reference')
    assert torch.autograd.grad(legacy, phi, retain_graph=True)[0] < 0
    assert torch.autograd.grad(safe, phi, retain_graph=True)[0] > 0
    safe.backward(); assert reference.grad is None


def test_tree_rgb_gradient_is_exactly_preserved():
    rgb = torch.full((3, 1, 2), .6, requires_grad=True)
    target = torch.full_like(rgb, .3); reference = torch.full_like(rgb, .8)
    regions = {'tree': torch.ones(1, 2, dtype=torch.bool),
               'rigid': torch.zeros(1, 2, dtype=torch.bool), 'sky': torch.zeros(1, 2, dtype=torch.bool)}
    gradients = [torch.autograd.grad(candidate_rgb_loss(rgb, target, reference, regions,
                 noncanopy_target=mode), rgb, retain_graph=True)[0]
                 for mode in ('legacy_ground_truth', 'source_reference')]
    assert torch.equal(*gradients)
