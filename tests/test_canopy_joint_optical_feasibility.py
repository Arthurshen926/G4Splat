import numpy as np
import pytest
from scripts.canopy_joint_optical_feasibility import solve_optical_intervals


def test_single_ray_possible_shared_wall_ray_impossible():
    single = solve_optical_intervals([[.8]], [np.log(10)], [np.inf], [10.])
    assert single['feasible_within_1e_7']
    joint = solve_optical_intervals([[.8], [.4]], [np.log(10), 0], [np.inf, -np.log(.95)], [10.])
    assert not joint['feasible_within_1e_7'] and joint['weighted_slack'] > 1


def test_selective_support_restores_joint_feasibility():
    result = solve_optical_intervals([[.8, .8], [.4, 0]], [np.log(10), 0], [np.inf, -np.log(.95)], [10., 10.])
    assert result['feasible_within_1e_7']
    assert result['tau'][1] > 2


def test_bounds_distinguished_from_missing_support():
    assert not solve_optical_intervals([[1.]], [2.], [np.inf], [1.])['feasible_within_1e_7']
    assert solve_optical_intervals([[1.]], [2.], [np.inf], [3.])['feasible_within_1e_7']
    assert not solve_optical_intervals([[0.]], [2.], [np.inf], [30.])['feasible_within_1e_7']


def test_malformed_intervals_rejected():
    with pytest.raises(ValueError): solve_optical_intervals([[-1.]], [0.], [1.], [1.])
    with pytest.raises(ValueError): solve_optical_intervals([[1.]], [2.], [1.], [1.])
