import pytest
import torch
from scripts.canopy_rgb_feasibility import color_feasibility, nonnegative_color_floor


def test_background_too_bright_cannot_be_fixed_by_candidate_colors():
    black = torch.full((3, 1, 1), .7); white = black+.2
    out = color_feasibility(black, white, black+.1, black-.2, torch.ones(1, 1, dtype=torch.bool))
    assert out['unreachable_color_fraction'] == 1
    assert out['minimum_color_only_mse'] == pytest.approx(.04)
    assert out['irreducible_error_fraction'] == pytest.approx(4/9)


def test_reachable_color_has_zero_bound_and_invalid_probe_rejected():
    black = torch.zeros(3, 2, 2); white = black+.4; mask = torch.ones(2, 2, dtype=torch.bool)
    assert color_feasibility(black, white, black+.3, black+.2, mask)['minimum_color_only_mse'] == 0
    white[1] += .1
    with pytest.raises(ValueError): color_feasibility(black, white, black, black, mask)


def test_nonnegative_floor_only_bounds_overbright_residual_not_high_radiance():
    black = torch.tensor([.7, .1, .2]).reshape(3, 1, 1)
    target = torch.tensor([.2, .8, 1.]).reshape(3, 1, 1)
    out = nonnegative_color_floor(black, black+1, target, torch.ones(1, 1, dtype=torch.bool))
    assert out['minimum_nonnegative_color_mse'] == pytest.approx(.25/3)
    assert out['below_black_floor_fraction'] == 1
