import torch
from scripts.audit_foliage_color_recovery import all_direction_color_upper_bound


def test_direction_varying_sh_must_not_be_mistaken_for_dc_death():
    features = torch.zeros(2, 16, 3); features[:, 0] = -2.
    features[1, 1] = 1.
    bound = all_direction_color_upper_bound(features)
    assert (bound[0] < 0).all()
    assert (bound[1] > 0).all()
