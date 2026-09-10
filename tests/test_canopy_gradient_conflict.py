import torch
from scripts.audit_canopy_rgb_gradient_conflict import gradient_summary


def test_opposed_gradient_does_not_always_reverse_the_update():
    result = gradient_summary(torch.tensor([-2., -1., 1., 0.]), torch.tensor([1., 2., -5., 0.]))
    assert result['canopy_growth_rows'] == 2
    assert result['opposed_rows'] == 2
    assert result['reversed_rows'] == 1
    assert abs(result['reversed_canopy_gradient_mass_fraction'] - 1 / 3) < 1.e-6


def test_no_canopy_gradient_is_not_evidence_of_conflict():
    result = gradient_summary(torch.zeros(3), torch.ones(3))
    assert result['canopy_growth_rows'] == 0
    assert result['reversed_canopy_gradient_mass_fraction'] == 0
    assert result['cosine'] == 0
