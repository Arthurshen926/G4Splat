import torch
from outdoor.optical_mass_gradient import mass_preserving_scale_alpha


def test_mass_retraction_chain_rule_matches_finite_difference_and_preserves_alpha_gradient():
    log_scale = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
    logit = torch.tensor(-.8, dtype=torch.float64, requires_grad=True)
    alpha = logit.sigmoid(); area = (2*log_scale).exp()
    corrected = mass_preserving_scale_alpha(alpha, area)
    assert torch.equal(alpha, corrected)
    g_l, g_s = torch.autograd.grad(corrected, (logit, log_scale))
    torch.testing.assert_close(g_l, alpha*(1-alpha))
    mass = (-torch.log1p(-alpha)*area).detach()
    h = 1e-5
    f = lambda s: -torch.expm1(-mass/(2*s).exp())
    fd = (f(log_scale+h)-f(log_scale-h))/(2*h)
    torch.testing.assert_close(g_s, fd, rtol=1e-8, atol=1e-9)


def test_uncorrected_scale_descent_can_ascend_after_mass_compensation():
    # A projected Gaussian near its centre needs more alpha. Widening at fixed
    # peak alpha helps, but widening at fixed integrated mass does the opposite.
    s = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(.3, dtype=torch.float64)
    mass = -torch.log1p(-alpha)
    profile = lambda a, x: a*torch.exp(-.5*.5**2/torch.exp(2*x))
    raw_loss = (profile(alpha, s)-.7).square()
    raw_g = torch.autograd.grad(raw_loss, s)[0]
    corrected_alpha = mass_preserving_scale_alpha(alpha, torch.exp(2*s))
    corrected_g = torch.autograd.grad((profile(corrected_alpha, s)-.7).square(), s)[0]
    retracted_loss = lambda x: (profile(-torch.expm1(-mass/torch.exp(2*x)), x)-.7).square()
    assert raw_g < 0 < corrected_g
    assert retracted_loss(s-.001*raw_g) > raw_loss
    assert retracted_loss(s-.001*corrected_g) < raw_loss


def test_scale_authority_gate_and_no_grad_forward_are_preserved():
    s = torch.ones(2, dtype=torch.float64, requires_grad=True)
    gate = torch.tensor([0., 1.], dtype=torch.float64)
    gated = s.detach()+gate*(s-s.detach())
    a = torch.tensor([.2, .4], dtype=torch.float64, requires_grad=True)
    result = mass_preserving_scale_alpha(a, gated.exp())
    ga, gs = torch.autograd.grad(result.sum(), (a, s))
    assert torch.equal(ga, torch.ones_like(a))
    assert gs[0] == 0 and gs[1] < 0
    with torch.no_grad(): assert mass_preserving_scale_alpha(a, s.exp()) is a
