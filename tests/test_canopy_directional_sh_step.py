import torch
from scripts.canopy_directional_sh_step import rescale_directional_sh_step_


def test_actual_step_matches_separate_adam_learning_rates_and_preserves_unauthorized_rows():
    whole = torch.ones(2, 2, 3, requires_grad=True)
    dc = whole[:1, :1].detach().clone().requires_grad_()
    rest = whole[:1, 1:].detach().clone().requires_grad_()
    combined = torch.optim.Adam([whole], lr=.025, eps=1e-15)
    separate = torch.optim.Adam([dict(params=[dc], lr=.025), dict(params=[rest], lr=.025*.05)], eps=1e-15)
    eligible = torch.tensor([True, False]); untouched = whole[1].detach().clone()
    for step in range(5):
        before = whole[eligible, 1:].detach().clone()
        grad = torch.full_like(whole, .2+step*.1); grad[1] = 0
        whole.grad = grad; dc.grad = grad[:1, :1].clone(); rest.grad = grad[:1, 1:].clone()
        combined.step(); separate.step()
        rescale_directional_sh_step_(whole, before, eligible, .05)
        torch.testing.assert_close(whole[:1, :1], dc, rtol=0, atol=2e-7)
        torch.testing.assert_close(whole[:1, 1:], rest, rtol=0, atol=2e-7)
        assert torch.equal(whole[1], untouched)


def test_zero_multiplier_restores_directional_coefficients_exactly():
    features = torch.ones(2, 2, 3); rows = torch.tensor([True, False]); before = features[rows, 1:].clone()
    features[rows, 1:] += .2
    rescale_directional_sh_step_(features, before, rows, 0.)
    assert torch.equal(features, torch.ones_like(features))
