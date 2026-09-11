import numpy as np
from scipy import sparse

from scripts.canopy_restricted_joint_solution import select_columns


def test_subset_keeps_strong_positive_and_removable_negative_contributors():
    K = sparse.csr_matrix([[.8, .1, 0., 0.], [0., .2, .5, .1]])
    lo = np.array([1., 0.]); hi = np.array([np.inf, .05])
    ref = np.array([.1, .1, .6, .1]); caps = np.ones(4)*3
    before = ref.copy()
    active = select_columns(K, lo, hi, ref, caps, 1)
    assert active[0] and active[2]
    assert (K[:, ~active] @ ref[~active] <= hi).all()
    np.testing.assert_array_equal(ref, before)


def test_zero_support_ray_is_not_fabricated():
    K = sparse.csr_matrix([[0., 0.], [0., 1.]])
    active = select_columns(K, np.array([1., 0.]), np.array([np.inf, .1]),
                            np.array([.2, .05]), np.ones(2), 8)
    assert not active.any()
