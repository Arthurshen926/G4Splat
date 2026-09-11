import numpy as np
import pytest
from scripts.canopy_radiance_prefix_bounds import optical_prefix_lower_bound
from scripts.canopy_joint_optical_feasibility import solve_optical_intervals


def test_light_behind_depth_is_not_charged_to_prefix():
    assert optical_prefix_lower_bound([.02]*3,[.1]*3,[1.]*3)==0.
    assert optical_prefix_lower_bound([.4]*3,[.1]*3,[1.]*3)==pytest.approx(np.log(.3999/.13))


def test_behind_leaf_cannot_satisfy_foreground_prefix_constraint():
    lower=optical_prefix_lower_bound([.4]*3,[.1]*3,[1.]*3)
    answer=solve_optical_intervals([[1.,0.]],[lower],[np.inf],[.1,10.])
    assert answer['weighted_slack']==pytest.approx(lower-.1)


def test_screen_suffix_allowance_cannot_strengthen_bound():
    raw=optical_prefix_lower_bound([.4]*3,[.1]*3,[1.]*3,leakage_fraction=0.)
    corrected=optical_prefix_lower_bound([.4]*3,[.1]*3,[1.]*3)
    assert corrected<raw
    with pytest.raises(ValueError):optical_prefix_lower_bound([-.1]*3,[.1]*3,[1.]*3)
